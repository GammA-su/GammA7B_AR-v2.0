from gamma7b_train.config import ModelCfg, TrainCfg, DataCfg, RunCfg, ContractCfg, contract_hash

def test_contract_hash_stable():
    cfg = ContractCfg(model=ModelCfg(), train=TrainCfg(domain_tokens={"default":"<|text|>"}), data=DataCfg(tokenizer_model="x.model"), run=RunCfg(out_dir="out"))
    h1 = contract_hash(cfg)
    h2 = contract_hash(cfg)
    assert h1 == h2
