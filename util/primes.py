import os
import sys
import numpy as np
sys.path.append("..")
from common.helper import PickleUtil
from itertools import islice, count
from math import sqrt



class Primes:
    PRIME_LIST_DIR = "../primes/"
    fname = "prime_number_list.pkl"

    # Ref: https://stackoverflow.com/a/27946768
    @classmethod
    def is_prime(cls, n):
        if n < 2 or not isinstance(n, int):
            return False
        for number in islice(count(2), int(sqrt(n) - 1)):
            if n % number == 0:
                return False

        return True

    @classmethod
    def get_prime_list(cls):
        PickleUtil.check_create_dir(cls.PRIME_LIST_DIR)

        pathname = os.path.join(cls.PRIME_LIST_DIR, cls.fname)
        if os.path.exists(pathname):
            print("Prime pickle found on disk, loading...")
            prime_list = PickleUtil.read(pathname)
        else:
            print("No prime pickle found on disk, creating...")
            prime_list = cls.generate_prime_list()
            print("Saving to disk...")
            PickleUtil.write(prime_list, pathname)

        print("Obtained prime list")
        return prime_list

    @classmethod
    def generate_prime_list(cls, largest_number=10000):
        prime_list = list()
        for ii in range(largest_number + 1):
            if cls.is_prime(ii):
                prime_list.append(ii)
        return np.array(prime_list)

    @classmethod
    def print_closest_primes(cls, NN):
        assert isinstance(NN, int) and NN > 1
        if cls.is_prime(NN):
            print(f"Entered value {NN} is prime!")
        else:
            primes = cls.get_prime_list()
            bigger_prime_idx = np.where(NN < primes)[0]
            if len(bigger_prime_idx) == 0:
                err_msg = f"Entered value {NN} is larger than all primes in our generated list\n"
                err_msg += f"Please run function generate_prime_list with a larger value of `largest_number`"
                raise RuntimeError(err_msg)

            lower = primes[bigger_prime_idx[0] - 1]
            upper = primes[bigger_prime_idx[0]]

            print("\n\n" + "-"*10)
            print(f" Entered: {NN} is not prime ")
            print(f" Closest smaller prime: {lower} ")
            print(f" Closest larger prime: {upper} ")
            print("-" * 10, end="\n\n")


if __name__ == "__main__":
    Primes.print_closest_primes(21)
