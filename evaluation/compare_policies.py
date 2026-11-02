"""Reuse the paired diagnostic pipeline for off vs grounded_v1, custom JSON only."""
from evaluation.compare_protocols import main as compare


def main(argv=None):
    return compare(argv,comparison='policy')


if __name__=='__main__':main()
