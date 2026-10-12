import numpy as np
import pytest


def test_go_closure_uses_safe_edges_resolves_alias_and_ignores_obsolete(tmp_path):
    from src.external_go_data import load_go_ontology, propagate_go
    source=tmp_path/'go.obo'
    source.write_text('''format-version: 1.2
[Term]
id: GO:0000001
name: parent
[Term]
id: GO:0000002
alt_id: GO:0000009
is_a: GO:0000001 ! parent
[Term]
id: GO:0000003
relationship: part_of GO:0000002 ! child
relationship: regulates GO:0000004 ! unsafe
[Term]
id: GO:0000004
name: unrelated
[Term]
id: GO:0000005
is_obsolete: true
is_a: GO:0000001 ! parent
''')
    ontology=load_go_ontology(source)
    assert propagate_go(['GO:0000003'],ontology)=={'GO:0000001','GO:0000002','GO:0000003'}
    assert propagate_go(['GO:0000009','GO:0000005','GO:9999999'],ontology)=={'GO:0000001','GO:0000002'}
    reference=load_go_ontology(source,include_reference_relationships=True)
    assert propagate_go(['GO:0000003'],reference)=={'GO:0000001','GO:0000002','GO:0000003','GO:0000004'}


def test_external_holdouts_removed_unknowns_masked_and_duplicates_merged():
    from src.external_go_data import external_training_records
    rows=[dict(accession='a',sequence='AAA',terms={'GO:1'}),
          dict(accession='b',sequence='AAA',terms={'GO:2'}),
          dict(accession='c',sequence='CCC',terms={'GO:2'}),
          dict(accession='d',sequence='DDD',terms={'GO:3'})]
    mapping={0:'GO:1',1:'GO:2'}
    result=external_training_records(rows,mapping,forbidden_sequences={'CCC'},label_count=3)
    assert len(result)==1 and result[0]['sequence']=='AAA'
    np.testing.assert_array_equal(result[0]['targets'],[1,1,0])
    np.testing.assert_array_equal(result[0]['known'],[1,1,0])
    assert result[0]['accessions']==['a','b']


def test_go_cycle_rejected(tmp_path):
    from src.external_go_data import load_go_ontology,propagate_go
    path=tmp_path/'cycle.obo'
    path.write_text('[Term]\nid: GO:1\nis_a: GO:2\n[Term]\nid: GO:2\nis_a: GO:1\n')
    with pytest.raises(ValueError,match='cycle'):
        propagate_go(['GO:1'],load_go_ontology(path))


def test_mapping_requires_independent_fit_check_and_rejects_weak_match():
    from scipy import sparse
    from src.external_go_data import select_go_mapping
    y=np.tile([[1,0],[0,1]],(8,1)).astype(np.uint8)
    go=sparse.csr_matrix(np.c_[y[:,1],y[:,0],np.ones(len(y))])
    result=select_go_mapping(y,go,['GO:b','GO:a','GO:root'],np.arange(8),np.arange(8,16),
                            min_support=3,min_f1=.95,min_margin=.1)
    assert [r['go_id'] for r in result]==['GO:a','GO:b']
    assert all(r['accepted'] for r in result)
    go[-1,0]=0
    result=select_go_mapping(y,go,['GO:b','GO:a','GO:root'],np.arange(8),np.arange(8,16),
                            min_support=3,min_f1=.95,min_margin=.1)
    assert result[0]['accepted'] and not result[1]['accepted']
    with pytest.raises(ValueError,match='disjoint'):
        select_go_mapping(y,go,['GO:b','GO:a','GO:root'],np.arange(8),np.arange(8))


def test_swissprot_evidence_parser_excludes_automatic_not_and_incomplete_records(tmp_path):
    import gzip
    from src.external_go_data import iter_swissprot_evidence
    path=tmp_path/'reviewed.dat.gz'
    with gzip.open(path,'wt') as f:
        f.write('''ID   EXAMPLE Reviewed; 3 AA.
AC   P00001; P99999;
DR   GO; GO:1; F:activity; IDA:UniProtKB.
DR   GO; GO:2; F:activity; IEA:InterPro.
DR   GO; GO:3; NOT; IMP:UniProtKB.
DR   GO; GO:4; F:activity; TAS:PubMed.
SQ   SEQUENCE   3 AA;
     ACD
//
ID   AUTOMATIC Reviewed; 3 AA.
AC   P00002;
DR   GO; GO:2; F:activity; IEA:InterPro.
SQ   SEQUENCE   3 AA;
     EFG
//
''')
    rows=list(iter_swissprot_evidence(path))
    assert len(rows)==1 and rows[0]['accession']=='P00001'
    assert rows[0]['sequence']=='ACD' and rows[0]['terms']=={'GO:1','GO:4'}
    assert rows[0]['evidence_codes']=={'IDA','TAS'}


def test_positive_go_rules_allow_incomplete_recall_but_require_reliable_precision():
    from scipy import sparse
    from src.external_go_data import select_positive_go_rules
    # Supported GO positives identify a strict subset of the target label.
    y=np.tile(np.r_[np.ones(200),np.zeros(200)],2)[:,None].astype(np.uint8)
    g=np.zeros((800,2),np.float32)
    g[np.r_[np.arange(100),np.arange(400,500)],0]=1
    g[:,1]=1
    result=select_positive_go_rules(y,sparse.csr_matrix(g),['GO:subset','GO:root'],np.arange(400),np.arange(400,800))
    assert result[0]['accepted'] and result[0]['go_id']=='GO:subset'
    assert result[0]['check_recall']==.5 and result[0]['check_precision']==1.
    assert result[0]['check_precision_lower_95']>.95
    g[600:603,0]=1
    result=select_positive_go_rules(y,sparse.csr_matrix(g),['GO:subset','GO:root'],np.arange(400),np.arange(400,800))
    assert not result[0]['accepted']
