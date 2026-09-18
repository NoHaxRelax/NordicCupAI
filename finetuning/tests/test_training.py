"""Small CPU-only regression checks; no model downloads or pretrained weights."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finetune_qa as qa


class TinyQA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(loss=((self.weight - input_ids[:, 0] / 100.) ** 2).mean())


class TrainingTests(unittest.TestCase):
    def test_pair_separator_does_not_truncate_context_end(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel(
            {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'question': 4, 'word': 5},
            unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
            special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                     unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]')
        context = ' '.join(['word'] * 17)
        record = dict(id='last-word', question='question', context=context, is_impossible=False,
                      answers={'answer_start': [context.rfind('word')]}, answer_end=len(context))
        feats, _, _ = qa.encode(tok, [record], 20, 4, 'word', True)
        _, _, offsets = qa.encode(tok, [record], 20, 4, 'word', False)
        self.assertEqual(max(item[1] for row in offsets for item in row if item), len(context))
        self.assertTrue(any(start > 0 for start in feats['start_positions']))

    def test_partial_accumulation_matches_full_batches(self):
        # Five examples make one full group of four and one partial group.
        feats = {'input_ids': [[10], [20], [30], [40], [50]],
                 'start_positions': [0] * 5, 'end_positions': [0] * 5}
        records = [{'id': 'train', 'transcript_id': 'train'},
                   {'id': 'test', 'transcript_id': 'test'}]

        def run(batch, accum):
            model, steps = TinyQA(), []

            class CountingSGD(torch.optim.SGD):
                def step(self, closure=None):
                    steps.append(True)
                    return super().step(closure)

            args = SimpleNamespace(smoke=False, max_len=384, stride=128, target='word',
                                   model='local-test-only', grad_checkpointing=False,
                                   optim='adamw', lr=.1, batch=batch, accum=accum,
                                   epochs=1, warmup=0., no_progress=True, folds=1,
                                   seed=17, eval_batch=1, n_best=1, max_answer_tokens=10,
                                   null_threshold=0., dump_preds=False, save_dir=None)
            metrics = dict(accuracy=1., mean_tiou=1., score=1., tiou_when_yes=1.,
                           n_pred_yes=1, n_gold_yes=1)
            with patch.object(qa, 'encode', return_value=(feats, ['test'], [])), \
                 patch('transformers.AutoModelForQuestionAnswering.from_pretrained', return_value=model), \
                 patch('torch.optim.AdamW', side_effect=lambda params, **kw: CountingSGD(params, lr=kw['lr'])), \
                 patch.object(qa, 'infer', return_value=(np.zeros((1, 1)), np.zeros((1, 1)))), \
                 patch.object(qa, 'postprocess', return_value={}), \
                 patch.object(qa, 'score', return_value=metrics), redirect_stdout(io.StringIO()):
                qa.run_fold(0, ['train'], ['test'], records, {}, None, args,
                            torch.device('cpu'), None)
            return model.weight.item(), len(steps)

        accumulated = run(batch=1, accum=4)
        grouped = run(batch=4, accum=1)
        self.assertEqual(accumulated[1], 2)
        self.assertEqual(grouped[1], 2)
        self.assertAlmostEqual(accumulated[0], grouped[0], places=6)
        # A partial last minibatch inside an accumulation group gets its
        # correct sample weight too (2+2+1 compared with one batch of 5).
        self.assertAlmostEqual(run(2, 4)[0], run(5, 1)[0], places=6)

    def test_real_data_folds_keep_conversations_separate(self):
        records, _ = qa.load_data(qa.ROOT / 'data/qa_train.jsonl', qa.ROOT / 'data/contexts.json', False)
        self.assertEqual(len(records), 386)
        folds, _ = qa.make_folds({r['transcript_id'] for r in records}, 3, .1, 17)
        seen = set()
        for train, held in folds:
            self.assertFalse(set(train) & set(held))
            self.assertFalse(seen & set(held))
            seen.update(held)

    def test_fold_cannot_have_empty_training_set(self):
        with self.assertRaises(SystemExit):
            qa.make_folds(['one'], 1, .1, 17)


if __name__ == '__main__':
    unittest.main()
