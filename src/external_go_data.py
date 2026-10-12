"""Conservative GO propagation and isolated positive/unknown external records."""

from pathlib import Path

import numpy as np


# Matches the pinned DeepGOPlus/DeepGO-SE source. TAS/IC are curated,
# non-experimental categories and are retained as such in the provenance.
REFERENCE_EVIDENCE_CODES = frozenset({'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',
    'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'})


def iter_swissprot_evidence(path, accepted_codes=REFERENCE_EVIDENCE_CODES):
    """Read complete reviewed flat-file records and retain supported GO positives."""
    import gzip
    accession, sequence, terms, codes, in_sequence = '', [], set(), set(), False
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('ID   '):
                accession, sequence, terms, codes, in_sequence = '', [], set(), set(), False
            elif line.startswith('AC   ') and not accession:
                accession = line[5:].split(';')[0].strip()
            elif line.startswith('DR   GO; '):
                fields = [part.strip() for part in line[5:].split(';')]
                if len(fields) >= 4:
                    code = fields[3].split(':')[0]
                    if code in accepted_codes and not any(part.startswith('NOT') for part in fields):
                        terms.add(fields[1])
                        codes.add(code)
            elif line.startswith('SQ   '):
                in_sequence = True
            elif line.startswith('//'):
                if accession and sequence and terms:
                    yield dict(accession=accession, sequence=''.join(sequence), terms=terms,
                               evidence_codes=codes.copy())
                in_sequence = False
            elif in_sequence:
                residues = ''.join(line.split())
                if not residues.isalpha() or not residues.isupper():
                    raise ValueError('Invalid Swiss-Prot sequence record')
                sequence.append(residues)


def load_go_ontology(path, *, include_reference_relationships=False):
    terms, aliases, current = {}, {}, None
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line == '[Term]':
            current = {'parents': set(), 'obsolete': False, 'aliases': []}
        elif line.startswith('['):
            current = None
        elif current is not None:
            if line.startswith('id: '):
                identifier = line[4:].strip()
                terms[identifier] = current
            elif line.startswith('alt_id: '):
                current['aliases'].append(line[8:].strip())
            elif line.startswith('is_a: '):
                current['parents'].add(line[6:].split()[0])
            elif line.startswith('relationship: part_of '):
                current['parents'].add(line.split()[2])
            elif include_reference_relationships and line.startswith('relationship: '):
                current['parents'].add(line.split()[2])
            elif line == 'is_obsolete: true':
                current['obsolete'] = True
    for identifier, term in terms.items():
        if not term['obsolete']:
            for alias in term['aliases']:
                aliases[alias] = identifier
    return {'terms': terms, 'aliases': aliases, 'closure': {}}


def propagate_go(annotations, ontology):
    terms, aliases, cache = ontology['terms'], ontology['aliases'], ontology['closure']

    def visit(identifier, visiting):
        identifier = aliases.get(identifier, identifier)
        if identifier in visiting:
            raise ValueError('GO ontology cycle')
        if identifier in cache:
            return cache[identifier]
        term = terms.get(identifier)
        if term is None or term['obsolete']:
            return frozenset()
        values = {identifier}
        for parent in term['parents']:
            values.update(visit(parent, visiting | {identifier}))
        cache[identifier] = frozenset(values)
        return cache[identifier]

    result = set()
    for identifier in annotations:
        result.update(visit(identifier, set()))
    return result


def external_training_records(rows, mapping, *, forbidden_sequences, label_count):
    """Merge exact duplicates; absent GO annotations remain unknown, never negative."""
    by_sequence = {}
    for row in rows:
        sequence = row['sequence']
        if sequence in forbidden_sequences:
            continue
        indices = [int(column) for column, go in mapping.items() if go in row['terms']]
        if not indices:
            continue
        if any(column < 0 or column >= label_count for column in indices):
            raise ValueError('Mapped columns outside the original label space')
        if sequence not in by_sequence:
            by_sequence[sequence] = dict(sequence=sequence, accessions=[],
                targets=np.zeros(label_count, np.uint8), known=np.zeros(label_count, np.uint8))
        record = by_sequence[sequence]
        record['accessions'].append(str(row['accession']))
        record['targets'][indices] = 1
        record['known'][indices] = 1
    return list(by_sequence.values())


def select_go_mapping(targets, go_matrix, go_ids, fitting, checking, *,
                      min_support=20, min_f1=.95, min_margin=.02):
    """Choose GO semantics on fit rows and validate within a separate fit subset."""
    from scipy import sparse
    y = np.asarray(targets)
    fitting, checking = np.asarray(fitting), np.asarray(checking)
    if set(fitting) & set(checking) or not len(fitting) or not len(checking):
        raise ValueError('Mapping fitting and checking rows must be nonempty and disjoint')
    if (y.ndim != 2 or not np.isin(y, [0, 1]).all()
            or go_matrix.shape != (len(y), len(go_ids)) or len(go_ids) < 2):
        raise ValueError('Binary labels and aligned GO matrix required')
    matrix = sparse.csr_matrix(go_matrix, dtype=np.float32)

    def stats(indices):
        subset = matrix[indices]
        support = y[indices].sum(0).astype(np.float32)
        go_support = np.asarray(subset.sum(0)).ravel()
        overlap = (sparse.csr_matrix(y[indices].T, dtype=np.float32) @ subset).toarray()
        precision = overlap / np.maximum(go_support[None, :], 1)
        recall = overlap / np.maximum(support[:, None], 1)
        f1 = 2 * overlap / np.maximum(support[:, None] + go_support[None, :], 1)
        return support, f1, precision, recall

    fit_support, fit_f1, _, _ = stats(fitting)
    check_support, check_f1, check_precision, check_recall = stats(checking)
    result = []
    for column in range(y.shape[1]):
        order = np.argsort(-fit_f1[column], kind='stable')[:2]
        best, runner = map(int, order)
        margin = float(fit_f1[column, best] - fit_f1[column, runner])
        accepted = bool(min(fit_support[column], check_support[column]) >= min_support
            and min(fit_f1[column, best], check_f1[column, best]) >= min_f1 and margin >= min_margin)
        result.append(dict(column=column, go_id=go_ids[best], runner_up_go=go_ids[runner],
            fit_f1=float(fit_f1[column, best]), check_f1=float(check_f1[column, best]),
            check_precision=float(check_precision[column, best]), check_recall=float(check_recall[column, best]),
            fit_support=int(fit_support[column]), check_support=int(check_support[column]),
            fit_margin=margin, accepted=accepted))
    # Ambiguous many-to-one anonymous columns cannot silently share external targets.
    candidates = {}
    for row in result:
        if row['accepted']:
            candidates.setdefault(row['go_id'], []).append(row)
    for colliding in candidates.values():
        if len(colliding) > 1:
            for row in colliding:
                row['accepted'] = False
                row['rejection'] = 'multiple_anonymous_columns_match_same_GO'
    return result


def select_positive_go_rules(targets, go_matrix, go_ids, fitting, checking, *,
                            min_fit_positives=100, min_check_positives=80,
                            min_fit_precision=.99, min_check_precision=.98,
                            min_precision_lower=.95, min_recall=.20):
    """Validate a GO-implies-positive rule; do not claim complete label equivalence."""
    from scipy import sparse
    y = np.asarray(targets)
    fitting, checking = np.asarray(fitting), np.asarray(checking)
    if (not len(fitting) or not len(checking) or set(fitting) & set(checking)
            or y.ndim != 2 or not np.isin(y, [0, 1]).all()
            or go_matrix.shape != (len(y), len(go_ids))):
        raise ValueError('Aligned binary data and disjoint fit/check subsets required')
    matrix = sparse.csr_matrix(go_matrix, dtype=np.float32)
    fit_go, check_go = matrix[fitting], matrix[checking]
    fit_count = np.asarray(fit_go.sum(0)).ravel()
    check_count = np.asarray(check_go.sum(0)).ravel()
    fit_overlap = (sparse.csr_matrix(y[fitting].T, dtype=np.float32) @ fit_go).toarray()
    check_overlap = (sparse.csr_matrix(y[checking].T, dtype=np.float32) @ check_go).toarray()
    precision = fit_overlap / np.maximum(fit_count[None, :], 1)
    recall = fit_overlap / np.maximum(y[fitting].sum(0)[:, None], 1)
    eligible = (precision >= min_fit_precision) & (recall >= min_recall) & (fit_count >= min_fit_positives)
    rows = []
    for column in range(y.shape[1]):
        candidates = np.flatnonzero(eligible[column])
        if not len(candidates):
            rows.append(dict(column=column, accepted=False, reason='no_supported_fit_positive_rule'))
            continue
        # Coverage selects on fitting only. The check subset never chooses a replacement.
        best = int(candidates[np.argmax(fit_overlap[column, candidates])])
        n = float(check_count[best])
        p = float(check_overlap[column, best] / max(n, 1))
        check_recall = float(check_overlap[column, best] / max(y[checking, column].sum(), 1))
        z = 1.96
        lower = ((p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)) if n else 0.
        accepted = bool(n >= min_check_positives and p >= min_check_precision
                        and lower >= min_precision_lower and check_recall >= min_recall)
        rows.append(dict(column=column, go_id=go_ids[best], accepted=accepted,
            relation='supported_GO_implies_anonymous_positive', fit_precision=float(precision[column, best]),
            fit_recall=float(recall[column, best]), fit_rule_positives=int(fit_count[best]),
            check_precision=p, check_precision_lower_95=float(lower), check_recall=check_recall,
            check_rule_positives=int(n), external_absence_means_unknown=True))
    return rows
