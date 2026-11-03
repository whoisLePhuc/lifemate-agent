import unittest
try:
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
except ModuleNotFoundError:
    torch=None
from training.sparse_loss import sparse_action_loss


@unittest.skipIf(torch is None,'torch/transformers required for tiny-model loss equivalence')
class SparseTests(unittest.TestCase):
    def test_matches_full_loss_and_all_parameter_gradients(self):
        torch.manual_seed(7)
        config=Qwen3Config(vocab_size=41,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,tie_word_embeddings=False)
        model=Qwen3ForCausalLM(config).float().eval()
        ids=torch.randint(0,41,(2,12));labels=ids.clone();labels[0,:7]=-100;labels[1,:9]=-100;labels[1,-1]=-100
        batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
        full=model(**batch,use_cache=False).loss;full.backward()
        gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        sparse=sparse_action_loss(model,batch);sparse.backward()
        self.assertTrue(torch.allclose(full,sparse,atol=1e-6,rtol=1e-6))
        for name,param in model.named_parameters():
            if name in gradients:self.assertTrue(torch.allclose(gradients[name],param.grad,atol=2e-6,rtol=2e-5),name)
        self.assertGreater(model.model.embed_tokens.weight.grad.abs().sum(),0)
    def test_no_targets_fails_before_forward(self):
        ids=torch.ones(1,4,dtype=torch.long)
        with self.assertRaises(ValueError):sparse_action_loss(None,{'input_ids':ids,'attention_mask':ids,'labels':ids*-100})

    def test_inference_rejects_environment_adapter_before_loading(self):
        from pathlib import Path
        from lifemate.agent.planner import LocalPlanner
        with self.assertRaisesRegex(ValueError,'environment paths'):
            LocalPlanner(Path.cwd(),adapter_path=Path.cwd()/'forbidden_environment'/'adapter')


if __name__=='__main__':unittest.main()
