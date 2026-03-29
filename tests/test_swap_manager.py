from modules.swap_manager import SwapManager


def test_swap_create_fails_without_size(tmp_path):
    manager = SwapManager(swap_path=str(tmp_path / "swap"), swap_size_gb=None)
    assert manager.create_swap() is False

