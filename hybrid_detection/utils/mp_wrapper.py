
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def multiprocess_run(func, worker_args, n_workers):
    mp_context = mp.get_context("spawn")

    futures = []
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, max_tasks_per_child=1) as executor:
        try:
            futures = [executor.submit(func, *args) for args in worker_args]

            for future in tqdm(as_completed(futures), total=len(futures), desc='Completed batches'):
                try:
                    result = future.result()
                except Exception as exc:
                    errors.append(exc)

        except KeyboardInterrupt:
            logging.warning("Interrupted, terminating workers")

            for future in futures:
                future.cancel()

            if hasattr(executor, "terminate_workers"): # Python 3.14+
                executor.terminate_workers()
            else: # Older Python versions
                for process in getattr(executor, "_processes", {}).values():
                    process.terminate()

            raise

    if errors:
        raise Exception(f"{len(errors)} workers failed: {errors}")
