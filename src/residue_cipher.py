"""Recover a fixed residue permutation from complete external sequence matches."""
from collections import defaultdict


ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWY')


def pattern(sequence):
    symbols = {}
    return bytes(symbols.setdefault(a,len(symbols)) for a in sequence)


def decode_sequence(sequence, mapping):
    if len(set(mapping.values())) != len(mapping) or not set(mapping) <= ALPHABET or not set(mapping.values()) <= ALPHABET:
        raise ValueError('Residue mapping must be a standard amino-acid bijection')
    if any(a in ALPHABET and a not in mapping for a in sequence):
        raise ValueError('Mapping missing a standard residue')
    return sequence.translate(str.maketrans(mapping))


def infer_mapping(competition, external, fit_ids, *, min_matches=40, prefix_length=80):
    frame = competition.set_index('protein_id')
    if (not frame.index.is_unique or len(set(fit_ids)) != len(fit_ids)
            or not set(fit_ids) <= set(frame.index)):
        raise ValueError('Unique declared fit IDs must exist in competition data')
    index = defaultdict(list)
    for pid, sequence in frame.loc[fit_ids,'sequence'].items():
        if len(sequence) >= prefix_length:
            index[pattern(sequence[:prefix_length])].append((pid,sequence))
    mapping, matches, seen = {}, [], set()
    for accession, original in zip(external.Entry,external.Sequence):
        if not isinstance(original,str) or len(original) < prefix_length:
            continue
        for pid, encoded in index.get(pattern(original[:prefix_length]),[]):
            if len(encoded) != len(original) or pid in seen:
                continue
            proposed = {}
            valid = True
            for a,b in zip(encoded,original):
                if (a in proposed and proposed[a] != b) or (a not in ALPHABET and a != b):
                    valid = False
                    break
                if a in ALPHABET:
                    if b not in ALPHABET:
                        valid = False
                        break
                    proposed[a] = b
            if not valid or len(set(proposed.values())) != len(proposed):
                continue
            if any(a in mapping and mapping[a] != b for a,b in proposed.items()):
                raise ValueError('Conflicting full-sequence residue bijections')
            mapping.update(proposed)
            if len(set(mapping.values())) != len(mapping):
                raise ValueError('Conflicting full-sequence residue bijections')
            matches.append(dict(protein_id=str(pid),external_accession=str(accession),
                                residues_compared=len(original),covered_standard_residues=len(proposed)))
            seen.add(pid)
    if len(matches) < min_matches:
        raise ValueError(f'Need {min_matches} independent full-sequence matches; obtained {len(matches)}')
    if set(mapping) != ALPHABET or set(mapping.values()) != ALPHABET:
        raise ValueError('All 20 standard residues must be identified')
    # Recheck every accepted complete pair under the globally frozen mapping.
    originals = external.set_index('Entry').Sequence.to_dict()
    for row in matches:
        if decode_sequence(frame.loc[row['protein_id'],'sequence'],mapping) != originals[row['external_accession']]:
            raise ValueError('Complete decoded sequence mismatch')
    return dict(mapping=dict(sorted(mapping.items())),matches=matches,labels_used=False,
                fit_ids_only=True,min_matches=min_matches,prefix_length=prefix_length,
                full_sequences_verified=True)
