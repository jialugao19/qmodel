import polars as pl
import os
import fasteners  # type: ignore
import hashlib
# from loguru import logger
from typing import Optional
from logging import getLogger

logger = getLogger(__name__)


__all__ = ["TmpFeather"]

os.makedirs("/dev/shm/tmp", exist_ok=True)

def check_exists_shm(filename: str) -> bool:
    with fasteners.InterProcessLock(filename + ".lock", logger=logger):
        # check if the file exists in shared memory
        if not os.path.exists(filename):
            logger.debug(f"file {filename} does not exist in shm")
            return False
        else:
            logger.debug(f"file {filename} exists in shm")
            return True

def read_from_shm(filename: str) -> pl.DataFrame:

    cnt_filename = filename + ".cnt"
    lock_filename = filename + ".lock"
    with fasteners.InterProcessLock(lock_filename, logger=logger):
        assert os.path.exists(filename), f"file {filename} not found"
        assert os.path.exists(cnt_filename), f"file {cnt_filename} not found"

        # read the counter file
        df = pl.read_ipc(filename, memory_map=True)
        with open(cnt_filename, "r+") as f:
            cnt = int(f.read())
            f.seek(0)
            f.write(str(cnt + 1))
            f.truncate()
            logger.debug(f"increment counter for {filename} to {cnt+1}")
    return df


def save_disk_and_read_from_shm(disk_filename: str, filename: str) -> pl.DataFrame:
    # save the dataframe to shm file and read it from shared memory
    # safe for multiprocessing via fasteners process lock

    cnt_filename = filename + ".cnt"
    lock_filename = filename + ".lock"
    with fasteners.InterProcessLock(lock_filename, logger=logger):  # will create file if non-exists
        # if non-exist, create it
        if not os.path.exists(filename):
            df = pl.read_ipc(disk_filename, memory_map=False)
            df.write_ipc(filename, compression="uncompressed")
            logger.debug(f"write {filename} to shm")

        # if non-exist, create it and add counter
        if not os.path.exists(cnt_filename):
            with open(cnt_filename, "w") as f:
                f.write("0")

        # safe since the fasteners lock is locked per process and file
        df = read_from_shm(filename)

    return df

def save_and_read_from_shm(df: pl.DataFrame, filename: str) -> pl.DataFrame:
    # save the dataframe to shm file and read it from shared memory
    # safe for multiprocessing via fasteners process lock

    cnt_filename = filename + ".cnt"
    lock_filename = filename + ".lock"
    with fasteners.InterProcessLock(lock_filename, logger=logger):  # will create file if non-exists
        # if non-exist, create it
        if not os.path.exists(filename):
            df.write_ipc(filename, compression="uncompressed")
            logger.debug(f"write {filename} to shm")

        # if non-exist, create it and add counter
        if not os.path.exists(cnt_filename):
            with open(cnt_filename, "w") as f:
                f.write("0")

        # safe since the fasteners lock is locked per process and file
        df = read_from_shm(filename)

    return df


def close_shm_file(filename: str) -> int:
    # reduce the ref counter

    cnt_filename = filename + ".cnt"
    lock_filename = filename + ".lock"
    to_remove = False
    with fasteners.InterProcessLock(lock_filename, logger=logger):  # will create file if non-exists
        # read the counter file
        assert os.path.exists(filename), f"file {filename} not found"
        with open(cnt_filename, "r+") as f:
            cnt = int(f.read()) - 1
            if cnt == 0:
                # if counter is 0, delete the file
                os.remove(filename)
                to_remove = True
                logger.debug(f"delete {filename} from shm")
            else:
                # otherwise, decrement the counter
                f.seek(0)
                f.write(str(cnt))
                f.truncate()
                logger.debug(f"decrement counter for {filename} to {cnt}")
        if to_remove:
            os.remove(cnt_filename)
        return cnt


def hash_filename(filename: str) -> str:
    # hash the filename to a fixed length
    hashed = hashlib.md5(filename.encode()).hexdigest()
    return hashed[:16]

class NoDirectInitMeta(type):
    """
    Prevents direct instantiation of classes using this metaclass.
    Enforces usage of factory classmethods.
    Define these classes by setting the metaclass to this one.
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot instantiate class {cls} directly, use classmethods that calls ._construct instead")

    # becomes available in classes, as classmethod
    def _construct(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

class TmpFeather(metaclass=NoDirectInitMeta):
    """Note: filename shall always be unique to prevent collision"""
    _df: pl.DataFrame
    _origin_filename: str
    _hashed_filename: str
    _folder = "/dev/shm/tmp"

    def __init__(self, df: pl.DataFrame, origin: str, hashed: str):
        self._df = df
        self._origin_filename = origin
        self._hashed_filename = hashed

    @classmethod
    def get_shm_filename(cls, raw_filename: str) -> str:
        return os.path.join(cls._folder, hash_filename(raw_filename))

    @classmethod
    def exists_in_shm(cls, raw_filename: Optional[str] = None, shm_filename: Optional[str] = None) -> bool:
        assert raw_filename is not None or shm_filename is not None
        if raw_filename is not None:
            shm_filename = os.path.join(cls._folder, hash_filename(raw_filename))

        assert shm_filename is not None
        return check_exists_shm(shm_filename)

    @classmethod
    def from_disk_file(cls, filename: str):
        """read from disk feather file"""
        origin = filename
        hashed = os.path.join(cls._folder, hash_filename(filename))
        df = save_disk_and_read_from_shm(filename, hashed)

        return cls._construct(df, origin, hashed)

    @classmethod
    def from_df(cls, df: pl.DataFrame, name: str):
        # ensure the name is unique

        # save the dataframe to shm file and read it from shared memory
        # safe for multiprocessing via fasteners process lock
        hashed_filename = os.path.join(cls._folder, hash_filename(name))
        os.makedirs(cls._folder, exist_ok=True)

        df = save_and_read_from_shm(df, hashed_filename)
        return cls._construct(df, name, hashed_filename)

    @classmethod
    def from_shm_file(cls, filename: str):
        name = ""
        df = read_from_shm(filename)
        return cls._construct(df, name, filename)

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def filename(self) -> str:
        return self._hashed_filename

    def close(self) -> int:
        # reduce counter and delete the file if counter is 0
        return close_shm_file(self._hashed_filename)
