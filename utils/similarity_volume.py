import numpy as np
import torch
import os, sys, time
import itertools
import warnings
import matplotlib.pyplot as plt

class SimVolume():
    def __init__(self, cosine_similarities) -> None:
        # e x m x 1
        aug = np.ones((cosine_similarities.shape[0], cosine_similarities.shape[1] + 1), dtype=np.float16)
        aug[:, :-1] = cosine_similarities

        self.aug = aug

    """
    # construct an m x m x... x m (e times) volume, where each entry is the product of all
    # cosine similarities row-wise.
    # To query the total of assignment (0,3), (1,4), (2,0),
    #   retrieve vol[3, 4, 0]

    To query an assignment with missing indices, like (0,3), (2, 4), (3,0)
    replace all missing indices with a -1
        eg. vol[3, -1, 4, 0]
    """
    def construct_volume(self):
        start = time.time()

        if self.aug.shape[0] < 2:
            print("Too few detected embs")
            return self.aug

        # prepare main volume
        # einsum_prompt = ""
        # for i, row in enumerate(self.aug[:-1]):
        #     einsum_prompt += chr(i + 97) + ","
        # einsum_prompt += chr(len(self.aug) + 97 - 1)
        # print("prompt: ", einsum_prompt)

        # t = [row for row in self.aug]
        # volume = np.einsum(einsum_prompt, *t)

        volume = np.einsum('i,j', self.aug[0], self.aug[1])
        for i, row in enumerate(self.aug[2:]):
            # print(i, row)
            vb = volume.shape
            volume = np.einsum('...i,j', volume, row)
            # print(f"einsum {i} done in {time.time() - start} seconds")
            # print(f"{vb} -> {volume.shape}\n")


        # disallow repeats
        mask = -np.inf * np.ones_like(volume)

        vol_dim = len(volume.shape)

        # unassigned may be repeated

        # print(f"e constructed at {time.time() - start}")
        perm_list = [i for i in itertools.permutations([i for i in range(volume.shape[0] - 1)], vol_dim)]
        # print(f"perms constructed at {time.time() - start}")
        # print(f"Len perms: {len(perm_list)}")

        # set unique assignments to 1
        for comb in perm_list:
            mask[comb] = 0

            for j in range(1, 1 << vol_dim):
                c = list(comb)

                # iterate through all binary choices of assigning or not assinging 
                unassigned = [k for k in range(vol_dim) if j & 1 << k]
                for u in unassigned:
                    c[u] = -1
                # print(c)

                mask[tuple(c)] = 0

        # atlesat one object must be assigned
        mask[list([-1 for i in range(vol_dim+1)])] = -np.inf
        
        # print(f"unique at {time.time() - start}")

        # apply mask, fix all 0 * np.infs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            rep_volume = volume + mask
            # print(f"mask at {time.time() - start}")
            rep_volume[np.isnan(rep_volume)] = -np.inf
            # print(f"fill at {time.time() - start}")

        return volume, rep_volume

    """
    Only calculate the edges of an actual volume, filling in some objects as unassigned
    Calculate nCs subvolumes, store them all in this object
    """
    def fast_construct_volume(self, subvolume_size):
        self.subvolumes = []

        if self.aug.shape[0] == 1:
            self.chosen_objects = [[0]]
            vol = self.aug[0]
            vol[-1] = -np.inf
            self.subvolumes.append(self.aug[0])
            return
        
        assert self.aug.shape[0] >= subvolume_size

        self.chosen_objects = [i for i in itertools.combinations([j for j in range(self.aug.shape[0])], subvolume_size)]
        for chosen in self.chosen_objects:
            # print(chosen)

            sub_aug = self.aug[list(chosen)]    # pick out the rows of self.aug that are in chosen
            
            volume = np.einsum('i,j', sub_aug[0], sub_aug[1])
            for i, row in enumerate(sub_aug[2:]):
                volume = np.einsum('...i,j', volume, row)

            # disallow repeats
            mask = -np.inf * np.ones_like(volume)

            vol_dim = len(volume.shape)

            # unassigned may be repeated

            # print(f"e constructed at {time.time() - start}")
            perm_list = [i for i in itertools.permutations([i for i in range(volume.shape[0] - 1)], vol_dim)]
            # print(f"perms constructed at {time.time() - start}")
            # print(f"Len perms: {len(perm_list)}")

            # set unique assignments to 1
            for comb in perm_list:
                mask[comb] = 0

                for j in range(1, 1 << vol_dim):
                    c = list(comb)

                    # iterate through all binary choices of assigning or not assinging 
                    unassigned = [k for k in range(vol_dim) if j & 1 << k]
                    for u in unassigned:
                        c[u] = -1
                    # print(c)

                    mask[tuple(c)] = 0
        
            # atlesat one object must be assigned
            mask[list([-1 for i in range(vol_dim+1)])] = -np.inf

            # apply mask, fix all 0 * np.infs
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
                rep_volume = volume + mask
                # print(f"mask at {time.time() - start}")
                rep_volume[np.isnan(rep_volume)] = -np.inf
                # print(f"fill at {time.time() - start}")

            self.subvolumes.append(rep_volume)

    """
    Simpler volume, no space for rearrangements, all given objects are given an assignment
    """
    def construct_volume_choose_e(self, chosen_e):
        assert len(chosen_e) <= self.aug.shape[0]

        volume = np.einsum('i,j', self.aug[chosen_e[0]], self.aug[chosen_e[1]])
        for i, row_num in enumerate(chosen_e[2:]):
            # print(i, row)
            row = self.aug[row_num]
            vb = volume.shape
            volume = np.einsum('...i,j', volume, row)
            # print(f"einsum {i} done in {time.time() - start} seconds")
            # print(f"{vb} -> {volume.shape}\n")
        return volume
    
    def get_top_indices(self, vol, k):
        top_k = []
        for i in range(k):
            ind = np.unravel_index(
                np.argmax(vol, axis=None),
                vol.shape
            )

            # print(f"ind: {ind} | val: {vol[ind]}")
            top_k.append([ind, vol[ind]])
            # replace with -inf to look for the second highest
            vol[ind] = -np.inf
        return top_k

    def conv_coords_to_pairs(self, vol, coords):
        assns = []
        unassigned_ind = vol.shape[0] - 1
        for c, cost in coords:
            assn = []
            for i, c_i in enumerate(c):
                if c_i != unassigned_ind:
                    assn.append([i, c_i])
            
            if len(assn) == 0:
                continue
            
            assns.append([assn, cost])
        return assns

    # search across all generated subvolumes for their topk costs, sort for the top k costs across all subvolumes
    # convert to assignments after sorting
    def get_top_indices_from_subvolumes(self, num_per_length=3):
        top_k = []
        # assume the top k is split equally amongst all lengths up to num_detected
        k = num_per_length * self.aug.shape[0] * 4      # take 4 times more assns just in case the above 
        for chosen, subvol in zip(self.chosen_objects, self.subvolumes):
            for i in range(k):
                ind = np.unravel_index(
                    np.argmax(subvol, axis=None),
                    subvol.shape
                )

                top_k.append([chosen, ind, subvol[ind]])
                subvol[ind] = -np.inf
        
        # get global top k assignments and convert to assignments
        all_filtered_topk = []
        assns = []

        print("\nshapes")
        for v in self.subvolumes:
            print(v.shape)
        print()

        unassigned_ind = self.subvolumes[0].shape[0] - 1
        for coords in top_k:
            
            filtered = []
            for i, c_i in zip(coords[0], coords[1]):
                if c_i != unassigned_ind:
                    filtered.append([i, c_i])
            
            if len(filtered) == 0:
                continue
            
            if filtered not in assns:
                assns.append(filtered)

                all_filtered_topk.append([filtered, coords[2]])
        
        filtered_topk = []
        for i in range(1, self.aug.shape[0] + 1):
            print("Length: ", i)
            correct_length = [f for f in all_filtered_topk if len(f[0]) == i]
            correct_length = sorted(correct_length, key= lambda x: x[-1], reverse=True)[:num_per_length]
            filtered_topk += correct_length

        assns = [a[0] for a in filtered_topk]

        # remove duplicates if any
        # loop to 
        no_dupe_assns = []
        for a in assns:
            if type(a) == list:
                  hashable_a = tuple(a)
            else: hashable_a = a

            if hashable_a not in no_dupe_assns:
                no_dupe_assns.append(a)

        assns = no_dupe_assns

        return assns



class TestSimVolume():
    # use to check volume
    def test_vol(vol, cs, verbose=False):
        num = cs.shape[0]
        indices = tuple([np.random.randint(0,num) for i in range(num)])

        prod = 1
        for n, i in enumerate(indices):
            prod *= cs[n,i]

        if verbose:
            print(f"Indices: {indices}")
            print(f"Volume says: {vol[indices]}")
            print(f"Product: {prod}")

        return vol[indices] == prod

    # use for volume
    def test_missing(vol, cs, verbose=False):
        num = cs.shape[0]
        indices = [np.random.randint(0,num) for i in range(num)]

        random_missing_idx = np.random.randint(0,num)
        indices[random_missing_idx] = -1
        indices = tuple(indices)

        prod = 1
        for n, i in enumerate(indices):
            if i == -1:
                continue
            prod *= cs[n,i]

        if verbose:
            print(f"Indices: {indices}")
            print(f"Missing assingment: {random_missing_idx}")
            print(f"Volume says: {vol[indices]}")
            print(f"Product should be zero: {prod}")

        return vol[indices] == prod

    # use for rep vol
    def test_repeated(rep_vol, cs, verbose=False):
        num = cs.shape[0]
        indices = tuple([np.random.randint(0,num) for i in range(num)])

        if len(set(indices)) != len(indices):
            prod = -np.inf
        else:
            prod = 1
            for n, i in enumerate(indices):
                prod *= cs[n,i]

        if verbose:
            print(f"Indices: {indices}")
            print(f"Volume says: {rep_vol[indices]}")
            print(f"Product: {prod}")

        return rep_vol[indices] == prod

    # use for rep vol
    def test_repeated_missing(rep_vol, cs, verbose=False):
        num = cs.shape[0]
        indices = [np.random.randint(0,num) for i in range(num)]

        random_missing_idx = np.random.randint(0,num)
        indices[random_missing_idx] = -1
        indices = tuple(indices)


        if len(set(indices)) != len(indices):
            prod = -np.inf
        else:
            prod = 1
            for n, i in enumerate(indices):
                if i == -1:
                    continue
                prod *= cs[n,i]

        if verbose:
            print(f"Indices: {indices}")
            print(f"Volume says: {rep_vol[indices]}")
            print(f"Product: {prod}")

        return rep_vol[indices] == prod

    def test_repeated_multiple_missing(rep_vol, cs, verbose=False):
        num = cs.shape[0]
        indices = [np.random.randint(0,num) for i in range(num)]

        random_missing_idx1 = np.random.randint(0,num)
        random_missing_idx2 = np.random.randint(0,num)
        indices[random_missing_idx1] = -1
        indices[random_missing_idx2] = -1
        indices = tuple(indices)

        if len(set(indices))+1 != len(indices) and random_missing_idx1 != random_missing_idx2:
            prod = -np.inf
        elif len(set(indices)) != len(indices) and random_missing_idx1 == random_missing_idx2:
            prod = -np.inf
        else:
            prod = 1
            for n, i in enumerate(indices):
                if i == -1:
                    continue
                prod *= cs[n,i]

        if verbose:
            print(f"Indices: {indices}")
            print(f"Volume says: {rep_vol[indices]}")
            print(f"Product: {prod}")

        return rep_vol[indices] == prod

# plot a graph of time vs complexity
def plot_time_graphs():
    time_val = []
    n_k = []
    for n_i in range(5):
        row = []
        for k_j in range(50):
            r = np.random.rand(n_i, k_j)
            sv = SimVolume(r)

            start = time.time()
            fast_construct_volume(sv.aug)
            time_taken = time.time() - start

            print(f"{n_i} x {k_j} took {time_taken} seconds")
            row.append(time_taken)
        time_val.append(row)

        # if n_i >= 4:
        plt.figure()
        for row_num, y in enumerate(time_val):
            plt.plot([i for i in range(50)], np.log10(np.array(y)), label=f"N = {row_num}")
        plt.legend()
        plt.savefig(f"./simvol_plots/with_numba_upto_{row_num}")
        plt.close()
        times = np.array(time_val)
        np.save(f"simvol_plots/with_numba_times_upto_{n_i}.npy", times)
        np.save(f"simvol_plots/with_numba_log10_times_upto_{n_i}.npy", np.log10(times))

        print(n_i, "done!")

if __name__ == "__main__":

    cs = np.array([[i for i in range(10)]])
    cs2 = np.array([i for i in range(4)])
    cs = cs.reshape(-1,1) + cs2.reshape(1,-1)

    # r = np.random.rand(*cs.shape)
    print(f"cs shape: {cs.shape}")

    sv = SimVolume(cs)
    tsv = TestSimVolume()

    start = time.time()
    
    # _, rep_vol = sv.construct_volume()
    # topk_full = sv.get_top_indices(rep_vol, 10)
    # assns = sv.conv_coords_to_pairs(rep_vol, topk_full)
    # print(f"Full volume: {assns}")

    sv.fast_construct_volume(3)
    print(sv.subvolumes)
    print(sv.chosen_objects)

    topk_sub = sv.get_top_indices_from_subvolumes(3)
    print(f"top: {topk_sub}")
    
    print()
    for i in (topk_sub):
        print(f"{i}\t")

    # time_taken = time.time() - start
    # print(f"in {time_taken} seconds")
    # start = time.time()

    # plot_time_graphs()

    exit(0)

    for i in range(100):
        assert(tsv.test_vol(vol, cs, False))
    print("Basic test passed")

    for i in range(100):
        assert(tsv.test_missing(vol, cs, False))
    print("Missing assignment test passed")

    for i in range(100):
        assert(tsv.test_repeated(rep_vol, cs, False))
    print("Repeated assignment test passed")

    for i in range(100):
        assert(tsv.test_repeated_missing(rep_vol, cs, False))
    print("Repeated missing assignment test passed")

    for i in range(100):
        assert(tsv.test_repeated_multiple_missing(rep_vol, cs, False))
    print("Repeated multiple missing assignment test passed")
    
    """
    Indices: (2, 4, 3, 0, -1)
    Volume says: 150.0
    Product: 1200
    """

    time_taken = time.time() - start
    print(f"Tests completed in {time_taken} seconds")