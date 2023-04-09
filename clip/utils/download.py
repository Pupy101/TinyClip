import hashlib
import ssl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, TypeVar, Union

import requests
from tqdm import tqdm

from clip.types import DownloadFile

Item = TypeVar("Item")


def configure_ssl() -> None:
    try:
        # W0212=protected-access
        unverified_https_context = ssl._create_unverified_context  # pylint: disable=W0212
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = unverified_https_context  # pylint: disable=W0212


def md5(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.md5(data).hexdigest()


def generator_chunks(items: Iterable[Item], chunk_size: int) -> Generator[List[Item], None, None]:
    chunk: List[Item] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def download_file(item: DownloadFile) -> Tuple[str, Optional[Path]]:
    response = requests.get(item.url, stream=True, verify=False, timeout=1000)
    if response.status_code != 200:
        return item.url, None
    path = Path(item.dir) / f"{md5(item.url)}.jpeg"
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    return item.url, path


def download_files_th(
    items: List[DownloadFile], n_threads: int, tqdm_off: bool = False
) -> Dict[str, Path]:
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        result = list(tqdm(executor.map(download_file, items), total=len(items), disable=tqdm_off))
    return {k: v for k, v in result if v}


def download_files_mp(
    items: List[DownloadFile], n_pools: int, tqdm_off: bool = False
) -> Dict[str, Path]:
    with Pool(processes=n_pools) as pool:
        result = list(tqdm(pool.map(download_file, items), total=len(items), disable=tqdm_off))
    return {k: v for k, v in result if v}


def download_files_mp_th(
    items: List[DownloadFile],
    n_pools: int,
    n_threads: int,
    thread_chunk: int,
    tqdm_off: bool = False,
) -> Dict[str, Path]:
    func = partial(download_files_th, n_threads=n_threads, tqdm_off=True)
    with Pool(processes=n_pools) as pool:
        results = list(
            tqdm(
                pool.map(func, generator_chunks(items, thread_chunk)),
                total=round(len(items) / thread_chunk),
                disable=tqdm_off,
            )
        )
    overall_result: Dict[str, Path] = {}
    for result in results:
        overall_result |= result
    return overall_result
