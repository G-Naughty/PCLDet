import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
#from ..obb.obb_base_sampler import OBBBaseSampler
from ..rotate_random_sampler import RRandomSampler

@BBOX_SAMPLERS.register_module()
class OBBCateBalanceSampler(RRandomSampler):
    """Random sampler

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 num_class=37,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(OBBCateBalanceSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                               add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))
        self.num_cls = num_class
    def random_category_balance_choice(self, assign_result,gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        pos_labels = assign_result.labels[gallery]
        #all_labels = torch.tensor([i for i in range(0, self.num_cls)]).reshape(-1,1)
        #label_idxes = pos_labels.eq(all_labels)
        categorys_id = []
        for i in range(0, self.num_cls):
            idx = torch.nonzero((pos_labels == i), as_tuple=False).flatten()
            if len(idx)> 0:
                categorys_id.append(gallery[idx])
        categorys_num = len(categorys_id)
        per_cate_num_expect = int(num//categorys_num)+1
        for j in range(categorys_num):
            if len(categorys_id[j]) > per_cate_num_expect:
                is_tensor = isinstance(categorys_id[j], torch.Tensor)
                if not is_tensor:
                    categorys_id[j] = torch.tensor(
                        categorys_id[j], dtype=torch.long, device=torch.cuda.current_device())
                prem = torch.randperm(categorys_id[j].numel(), device=gallery.device)
                unsample = categorys_id[j][prem[per_cate_num_expect:]].reshape(-1, 1)
                categorys_id[j] = categorys_id[j][prem[:per_cate_num_expect]]

                try:
                    unrand_inds = torch.cat([unrand_inds, unsample], dim=0)
                except:
                    unrand_inds = unsample
            categorys_id[j] = categorys_id[j].reshape(-1, 1)
            if j==0:
                rand_inds = categorys_id[j]
            else:
                rand_inds= torch.cat([rand_inds,categorys_id[j]],dim=0)

        rand_inds = rand_inds.flatten()
        if rand_inds.numel() < num:
            unrand_inds = unrand_inds.flatten()
            resample_num = num - rand_inds.numel()
            reprem = torch.randperm(unrand_inds.numel(), device=gallery.device)[:resample_num]
            rerand_inds = unrand_inds[reprem]
            rand_inds = torch.cat([rand_inds.reshape(-1, 1), rerand_inds.reshape(-1, 1)], dim=0)
            rand_inds = rand_inds.flatten()
        perm = torch.randperm(rand_inds.numel(), device=gallery.device)
        rand_inds = rand_inds[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_category_balance_choice(assign_result, pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)