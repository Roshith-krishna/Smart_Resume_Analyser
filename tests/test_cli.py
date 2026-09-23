"""Tests for CLI argument parsing (V1.2)."""

from src.cli import build_arg_parser


class TestCLIArgs:
    def test_default_top_k(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "test.pdf"])
        assert args.top_k == 5

    def test_custom_top_k(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "test.pdf", "--top-k", "10"])
        assert args.top_k == 10

    def test_default_candidate_pool_size(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "test.pdf"])
        assert args.candidate_pool_size == 50

    def test_custom_candidate_pool_size(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "test.pdf", "--candidate-pool-size", "100"])
        assert args.candidate_pool_size == 100

    def test_verbose_default_false(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "test.pdf"])
        assert args.verbose is False

    def test_verbose_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "test.pdf", "--verbose"])
        assert args.verbose is True

    def test_all_flags_together(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--resume", "test.pdf",
            "--top-k", "10",
            "--candidate-pool-size", "200",
            "--verbose",
            "--decimals", "6",
        ])
        assert args.top_k == 10
        assert args.candidate_pool_size == 200
        assert args.verbose is True
        assert args.decimals == 6
