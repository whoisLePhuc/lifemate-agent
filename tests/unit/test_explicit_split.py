"""A fixed inner split must partition outer train exactly and never cut a group."""
import unittest

from training.trajectories import explicit_inner_split

MANIFEST = {'train_ids': ['a', 'b', 'c', 'd', 'e'], 'dev_ids': ['x', 'y'],
            'groups': [['a', 'b'], ['c'], ['d', 'e'], ['x', 'y']]}


def supplied(**changes):
    base = {'train_ids': ['a', 'b', 'c'], 'validation_ids': ['d', 'e'], 'outer_dev_ids': ['x', 'y']}
    return {**base, **changes}


class ExplicitSplitTest(unittest.TestCase):
    def test_valid_split_is_normalised_to_the_derived_schema(self):
        s = explicit_inner_split(MANIFEST, supplied())
        self.assertEqual((s['train_ids'], s['validation_ids']), (['a', 'b', 'c'], ['d', 'e']))
        self.assertEqual((s['seen_ids'], s['forced_train_ids'], s['seed']), ([], [], None))
        self.assertEqual(s['groups'], [['a', 'b'], ['c'], ['d', 'e']])

    def test_rejects_overlap_missing_and_extra_ids(self):
        for bad in (supplied(validation_ids=['c', 'd', 'e']),
                    supplied(validation_ids=['d']),
                    supplied(validation_ids=['d', 'e', 'zz'])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                explicit_inner_split(MANIFEST, bad)

    def test_rejects_wrong_outer_dev(self):
        with self.assertRaises(ValueError):
            explicit_inner_split(MANIFEST, supplied(outer_dev_ids=['x']))

    def test_rejects_a_group_cut_across_the_split(self):
        with self.assertRaises(ValueError):
            explicit_inner_split(MANIFEST, supplied(train_ids=['a', 'c', 'd'], validation_ids=['b', 'e']))

    def test_rejects_duplicates_and_empty_lists(self):
        for bad in (supplied(train_ids=['a', 'a', 'b', 'c']), supplied(validation_ids=[])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                explicit_inner_split(MANIFEST, bad)


if __name__ == '__main__':
    unittest.main()
