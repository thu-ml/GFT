# Part of this code is modified from GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from functools import partial
from typing import Iterable, Optional, Tuple
import tensorflow.compat.v1 as tf
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import shutil
import torch
import time
import os
from torchmetrics.multimodal.clip_score import CLIPScore

resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """

    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


@torch.no_grad()
def compute_fid(fake_arr, gt_dir, device,
                resize_size=None, feature_extractor="inception",
                patch_fid=False):
    from coco_eval.cleanfid import fid
    center_crop_trsf = CenterCropLongEdge()

    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np)
        if patch_fid:
            # if image_pil.size[0] != 1024 and image_pil.size[1] != 1024:
            #     image_pil = image_pil.resize([1024, 1024])

            # directly crop to the 299 x 299 patch expected by the inception network
            if image_pil.size[0] >= 299 and image_pil.size[1] >= 299:
                image_pil = transforms.functional.center_crop(image_pil, 299)
            # else:
            #     raise ValueError("Image is too small to crop to 299 x 299")
        else:
            image_pil = center_crop_trsf(image_pil)

            if resize_size is not None:
                image_pil = image_pil.resize((resize_size, resize_size),
                                             Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError(
            "Unrecognized feature extractor [%s]" % feature_extractor)
    fid, fake_feats, real_feats = fid.compute_fid(
        # fid = fid.compute_fid(
        None,
        gt_dir,
        model_name=model_name,
        custom_image_tranform=resize_and_center_crop,
        use_dataparallel=False,
        device=device,
        pred_arr=fake_arr
    )
    # return fid
    return fid, fake_feats, real_feats


def evaluate_model(args, device, all_images, patch_fid=False):
    fid, fake_feats, real_feats = compute_fid(
        fake_arr=all_images,
        gt_dir=args.ref_dir,
        device=device,
        resize_size=args.eval_res,
        feature_extractor="inception",
        patch_fid=patch_fid
    )
    if patch_fid:
        return fid
    else:
        prec, recall = compute_prec_recall(real_feats, fake_feats)
        return fid, prec, recall
        # return fid, 0, 0


def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image


class CLIPScoreDataset(Dataset):
    def __init__(self, images, captions, transform, preprocessor) -> None:
        super().__init__()
        self.images = images
        self.captions = captions
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_pil = self.transform(image)
        image_pil = self.preprocessor(image_pil)
        caption = self.captions[index]
        return image_pil, caption


@torch.no_grad()
def compute_clip_score(
        images, captions, clip_model="ViT-B/32", device="cuda", how_many=30000):
    print("Computing CLIP score")
    import clip as openai_clip
    if clip_model == "ViT-B/32":
        clip, clip_preprocessor = openai_clip.load(
            "ViT-B/32", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-L/14":
        clip, clip_preprocessor = openai_clip.load(
            "ViT-L/14", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-G/14":
        import open_clip
        clip, _, clip_preprocessor = open_clip.create_model_and_transforms(
            "ViT-g-14", pretrained="laion2b_s12b_b42k")
        clip = clip.to(device)
        clip = clip.eval()
        clip = clip.float()
    else:
        raise NotImplementedError

    def resize_and_center_crop(image_np, resize_size=256):
        image_pil = Image.fromarray(image_np)
        image_pil = CenterCropLongEdge()(image_pil)
        if resize_size is not None:
            image_pil = image_pil.resize((resize_size, resize_size),
                                            Image.LANCZOS)
        return image_pil

    def simple_collate(batch):
        images, captions = [], []
        for img, cap in batch:
            images.append(img)
            captions.append(cap)
        return images, captions

    dataset = CLIPScoreDataset(
        images, captions, transform=resize_and_center_crop,
        preprocessor=clip_preprocessor
    )
    dataloader = DataLoader(
        dataset, batch_size=64,
        shuffle=False, num_workers=8,
        collate_fn=simple_collate

    )

    cos_sims = []
    count = 0
    # for imgs, txts in zip(images, captions):
    for index, (imgs_pil, txts) in enumerate(dataloader):
        # imgs_pil = [resize_and_center_crop(imgs)]
        # txts = [txts]
        # imgs_pil = [clip_preprocessor(img) for img in imgs]
        imgs = torch.stack(imgs_pil, dim=0).to(device)
        tokens = openai_clip.tokenize(txts, truncate=True).to(device)
        # Prepending text prompts with "A photo depicts "
        # https://arxiv.org/abs/2104.08718
        prepend_text = "A photo depicts "
        prepend_text_token = openai_clip.tokenize(prepend_text)[
            :, 1:4].to(device)
        prepend_text_tokens = prepend_text_token.expand(
            tokens.shape[0], -1)

        start_tokens = tokens[:, :1]
        new_text_tokens = torch.cat(
            [start_tokens, prepend_text_tokens, tokens[:, 1:]], dim=1)[:, :77]
        last_cols = new_text_tokens[:, 77 - 1:77]
        last_cols[last_cols > 0] = 49407  # eot token
        new_text_tokens = torch.cat(
            [new_text_tokens[:, :76], last_cols], dim=1)

        img_embs = clip.encode_image(imgs)
        text_embs = clip.encode_text(new_text_tokens)

        similarities = torch.nn.functional.cosine_similarity(
            img_embs, text_embs, dim=1)
        cos_sims.append(similarities)
        count += similarities.shape[0]
        if count >= how_many:
            break

    clip_score = torch.cat(cos_sims, dim=0)[:how_many].mean()
    clip_score = clip_score.detach().cpu().numpy()
    return clip_score


@torch.no_grad()
def compute_image_reward(
    images, captions, device
):
    import ImageReward as RM
    from tqdm import tqdm
    model = RM.load("ImageReward-v1.0", device=device,
                    download_root="/mnt/vepfs/base2/jiangkai/.cache/ImageReward")
    rewards = []
    for image, prompt in tqdm(zip(images, captions)):
        reward = model.score(prompt, Image.fromarray(image))
        rewards.append(reward)
    return np.mean(np.array(rewards))


@torch.no_grad()
def compute_diversity_score(
    lpips_loss_func, images, device
):
    # resize all image to 512 and convert to tensor
    images = [Image.fromarray(image) for image in images]
    images = [image.resize((512, 512), Image.LANCZOS) for image in images]
    images = np.stack([np.array(image) for image in images], axis=0)
    images = torch.tensor(images).to(device).float() / 255.0
    images = images.permute(0, 3, 1, 2)

    num_images = images.shape[0]
    loss_list = []

    for i in range(num_images):
        for j in range(i+1, num_images):
            image1 = images[i].unsqueeze(0)
            image2 = images[j].unsqueeze(0)
            loss = lpips_loss_func(image1, image2)

            loss_list.append(loss.item())
    return np.mean(loss_list)


class DistanceBlock:
    """
    Calculate pairwise distances between vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self, session):
        self.session = session

        # Initialize TF graph to calculate pairwise distances.
        with session.graph.as_default():
            self._features_batch1 = tf.placeholder(
                tf.float32, shape=[None, None])
            self._features_batch2 = tf.placeholder(
                tf.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf.cast(self._features_batch1, tf.float16),
                tf.cast(self._features_batch2, tf.float16),
            )
            self.distance_block = tf.cond(
                tf.reduce_all(tf.math.is_finite(distance_block_16)),
                lambda: tf.cast(distance_block_16, tf.float32),
                lambda: _batch_pairwise_distances(
                    self._features_batch1, self._features_batch2),
            )

            # Extra logic for less thans.
            self._radii1 = tf.placeholder(tf.float32, shape=[None, None])
            self._radii2 = tf.placeholder(tf.float32, shape=[None, None])
            dist32 = tf.cast(self.distance_block, tf.float32)[..., None]
            self._batch_1_in = tf.math.reduce_any(
                dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf.math.reduce_any(
                dist32 <= self._radii1[:, None], axis=0)

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.
        """
        return self.session.run(
            self.distance_block,
            feed_dict={self._features_batch1: U, self._features_batch2: V},
        )

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={
                self._features_batch1: batch_1,
                self._features_batch2: batch_2,
                self._radii1: radii_1,
                self._radii2: radii_2,
            },
        )


def compute_prec_recall(
    activations_ref: np.ndarray, activations_sample: np.ndarray
) -> Tuple[float, float]:
    manifold_estimator = ManifoldEstimator()
    radii_1 = manifold_estimator.manifold_radii(activations_ref)
    radii_2 = manifold_estimator.manifold_radii(activations_sample)
    pr = manifold_estimator.evaluate_pr(
        activations_ref, radii_1, activations_sample, radii_2
    )
    return (float(pr[0][0]), float(pr[1][0]))


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param session: the TensorFlow session.
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

        print("warming up TensorFlow...")
        # This will cause TF to print a bunch of verbose stuff now rather
        # than after the next print(), to help prevent confusion.
        self.warmup()

    def warmup(self):
        feats, radii = (
            np.zeros([1, 2048], dtype=np.float32),
            np.zeros([1, 1], dtype=np.float32),
        )
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros(
            [self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[
                    0: end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in _numpy_partition(distance_batch[0: end1 - begin1, :], seq, axis=1)
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(
                radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(self, features: np.ndarray, radii: np.ndarray, eval_features: np.ndarray):
        """
        Evaluate if new feature vectors are at the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = np.zeros(
            [self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros(
            [num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[
                    0: end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0: end1 -
                                                 begin1, :, None] <= radii
            batch_predictions[begin1:end1] = np.any(
                samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                radii[:, 0] / (distance_batch[0: end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(
                distance_batch[0: end1 - begin1, :], axis=1)

        return {
            "fraction": float(np.mean(batch_predictions)),
            "batch_predictions": batch_predictions,
            "max_realisim_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros(
            [len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros(
            [len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


def _batch_pairwise_distances(U, V):
    """
    Compute pairwise distances between two batches of feature vectors.
    """
    with tf.variable_scope("pairwise_dist_block"):
        # Squared norms of each row in U and V.
        norm_u = tf.reduce_sum(tf.square(U), 1)
        norm_v = tf.reduce_sum(tf.square(V), 1)

        # norm_u as a column and norm_v as a row vectors.
        norm_u = tf.reshape(norm_u, [-1, 1])
        norm_v = tf.reshape(norm_v, [1, -1])

        # Pairwise squared Euclidean distances.
        D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx: start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))
