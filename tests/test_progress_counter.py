from slpr.progress import ProgressCounter


def test_progress_counter_add_and_value():
    # Simulate a shared value with a local object implementing value/get_lock
    class Dummy:
        def __init__(self):
            self.value = 0

        def get_lock(self):
            # No-op context manager
            class CM:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return CM()

    shared = Dummy()
    pc = ProgressCounter(shared)
    pc.add(10)
    pc.add(5)
    assert shared.value == 15
