from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
import wave

from training.preparation import audit_audio, group_train, normalize, split_groups


def example(id, text, history=()):
    return NS(agent_input=NS(id=id, audio=Path(id)),
              reference=NS(audio_text=text, history_audio_text=history))


class PreparationTests(unittest.TestCase):
    def test_unicode_and_exact_text(self):
        self.assertEqual(normalize('  BẬT, ĐÈN! '), 'bật đèn')
        groups, _ = group_train([example('a','Bật đèn!'),example('b','bật đèn'),example('c','Tắt đèn')])
        self.assertEqual(groups, [['a','b'],['c']])
    def test_history_connection_and_transitive_groups(self):
        long = 'đặt cho tôi một chiếc xe về nhà'
        rows = [example('a',long),example('b','đổi địa chỉ', [long]),example('c','đổi địa chỉ')]
        self.assertEqual(group_train(rows)[0], [['a','b','c']])
    def test_audio_and_near_text_grouping(self):
        text = ' '.join(f'token{i}' for i in range(40))
        rows = [example('a',text),example('b',text+' thêm'),example('c','khác hoàn toàn'),example('d','nội dung khác')]
        groups, reasons = group_train(rows, {Path('a'):'1',Path('b'):'2',Path('c'):'3',Path('d'):'3'})
        self.assertEqual(groups, [['a','b'],['c','d']])
        self.assertGreater(reasons['current_near_trigram'], 0)
    def test_reproducible_disjoint_complete_and_grouped(self):
        groups = [['a','b'],['c'],['d'],['e'],['f']]
        first = split_groups(groups)
        self.assertEqual(first, split_groups(list(reversed(groups))))
        train,dev=map(set, first)
        self.assertFalse(train & dev)
        self.assertEqual(train|dev, set('abcdef'))
        self.assertEqual('a' in train, 'b' in train)
        for bad in [[['a','b']], [['a'],['a']]]:
            with self.assertRaises(ValueError): split_groups(bad)
    def test_grouping_input_order_independent(self):
        rows = [example('a','bật đèn'),example('b','Bật đèn!'),example('c','tắt đèn')]
        self.assertEqual(group_train(rows)[0], group_train(list(reversed(rows)))[0])
    def test_audio_decode_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); good=root/'good.wav'; bad=root/'bad.wav'
            with wave.open(str(good),'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0\0'*160)
            bad.write_bytes(good.read_bytes()[:-20])
            loader=NS(split_root=root,iter_inputs=lambda:iter([NS(audio=good,history=[{'role':'user','audio':bad}])]))
            report,digests=audit_audio(loader)
            self.assertEqual(report['decoded'],1)
            self.assertEqual(len(report['errors']),1)
            self.assertIn(good,digests)


if __name__=='__main__':unittest.main()
