import logging
import os
import os.path as osp 
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import confusion_matrix
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil
WANDB = True  
EMBEDDING_DIM = 128
NUM_INSTANCES = 12
EXPERIMENT_NAME = "DATOR_new_attempt1_rrc"
PROJECT_NAME = "ICRA"

def do_train_4DNet(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query,
             target_gpu):
    if WANDB:
        wandb.login(key="6ab81b60046f7d7f6a7dca014a2fcaf4538ff14a")
        wandb.init(project=PROJECT_NAME, name=EXPERIMENT_NAME)
    if os.path.exists(f"vis_positions"):
        shutil.rmtree(f"vis_positions")
    os.mkdir(f"vis_positions")
    if os.path.exists(f"vis"):
        shutil.rmtree(f"vis")
    os.mkdir(f"vis")
    if not os.path.exists(f"logs"):
        os.mkdir(f"logs")
    if os.path.exists(f"logs/{EXPERIMENT_NAME}"):
        shutil.rmtree(f"logs/{EXPERIMENT_NAME}")
    os.mkdir(f"logs/{EXPERIMENT_NAME}")
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = 20 
    eval_period = 5  

    device = "cuda"
    # epochs = cfg.SOLVER.MAX_EPOCHS
    epochs = 240  

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    # if device:
        # model.to(local_rank)
        # if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        #     print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for n_iter, (img, depth, vid, target_cam, target_view) in enumerate(loop):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            # img = img.to(device)
            # depth = depth.to(device)
            target = vid.to(target_gpu)
            # target_cam = target_cam.to(device)
            # target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, depth, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            # scaler.scale(loss).backward()
            # # nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            optimizer.step()

            # this condition is not met for Market dataset
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            loop.set_postfix(loss = loss_meter.avg, acc = acc_meter.avg.item(), lr = scheduler._get_lr(epoch)[0])
            torch.cuda.synchronize()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                save_path = osp.join("/ssd_scratch/cvit/vaibhav/ckpts/", EXPERIMENT_NAME, f"{epoch}.pth")  
                save_dir = osp.dirname(save_path) 
                os.makedirs(save_dir, exist_ok=True) 
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), save_path)
            else:
                save_path = osp.join("/ssd_scratch/cvit/vaibhav/ckpts/", EXPERIMENT_NAME, f"{epoch}.pth")  
                save_dir = osp.dirname(save_path) 
                os.makedirs(save_dir, exist_ok=True) 
                torch.save(model.state_dict(), save_path)

        if epoch % eval_period == 0:
            model.eval()
            batch_size = cfg.SOLVER.IMS_PER_BATCH
            embeddings = torch.empty((len(val_loader) * batch_size, EMBEDDING_DIM))
            count = 0
            evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator.reset()
            with torch.no_grad():
                for n_iter, (img, depth, vid, target_cam, target_view) in enumerate(val_loader):
                    img = img.to(device)
                    depth = depth.to(device)
                    target = vid.to(device)
                    target_cam = target_cam.to(device)
                    target_view = target_view.to(device)
                    img = torch.zeros(img.shape)
                    feat = model(img, depth, target, cam_label=target_cam, view_label=target_view )
                    feat = feat.cpu()
                    evaluator.update((feat, vid.cpu(), torch.tensor([0.0])))
                    # loss = loss_fn(score, feat, target, target_cam)
                    embeddings[n_iter * batch_size : n_iter * batch_size + img.shape[0]] = feat
                    count += img.shape[0]
                cmc, mAP, _, _, _, _, _ = evaluator.compute() 
                depth_mAP = mAP
                embeddings = embeddings[:count]
                # scores = F.cosine_similarity(embeddings.unsqueeze(0).repeat(count, 1, 1), embeddings.unsqueeze(1).repeat(1, count, 1))

                # print(f"computing the confusion matrix for depth-only case...")
                # scores = torch.zeros((count, count))
                # for i in tqdm(range(scores.shape[0])):
                #     for j in range(scores.shape[1]):
                #         scores[i][j] = embeddings[i] @ embeddings[j] / (torch.norm(embeddings[i]) * torch.norm(embeddings[j]))
                # positive_score = sum(torch.sum(scores[i : i + NUM_INSTANCES, i : i + NUM_INSTANCES]) for i in range(0, scores.shape[0], NUM_INSTANCES))
                # negative_score = torch.sum(scores) - positive_score
                # positive_score = positive_score / (count * count)
                # negative_score = negative_score / (count * count)
                scores = confusion_matrix(embeddings, embeddings) 
                plt.figure(figsize=(20, 20))
                plt.imshow(scores) 
                plt.colorbar() 
                plt.savefig(f"logs/{EXPERIMENT_NAME}/depth_{epoch}.jpg")


            batch_size = cfg.SOLVER.IMS_PER_BATCH
            embeddings = torch.empty((len(val_loader) * batch_size, EMBEDDING_DIM))
            count = 0
            evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator.reset()
            with torch.no_grad():
                for n_iter, (img, depth, vid, target_cam, target_view) in enumerate(val_loader):
                    img = img.to(device)
                    depth = depth.to(device)
                    target = vid.to(device)
                    target_cam = target_cam.to(device)
                    target_view = target_view.to(device)
                    depth = torch.zeros(depth.shape)
                    feat = model(img, depth, target, cam_label=target_cam, view_label=target_view )
                    feat = feat.cpu()
                    evaluator.update((feat, vid.cpu(), torch.tensor([0.0])))
                    # loss = loss_fn(score, feat, target, target_cam)
                    embeddings[n_iter * batch_size : n_iter * batch_size + img.shape[0]] = feat
                    count += img.shape[0]

                cmc, mAP, _, _, _, _, _ = evaluator.compute() 
                rgb_mAP = mAP
                embeddings = embeddings[:count]
                # scores = F.cosine_similarity(embeddings.unsqueeze(0).repeat(count, 1, 1), embeddings.unsqueeze(1).repeat(1, count, 1))

                # print(f"computing the confusion matrix for rgb-only case...")
                # scores = torch.zeros((count, count))
                # for i in tqdm(range(scores.shape[0])):
                #     for j in range(scores.shape[1]):
                #         scores[i][j] = embeddings[i] @ embeddings[j] / (torch.norm(embeddings[i]) * torch.norm(embeddings[j]))
                # positive_score = sum(torch.sum(scores[i : i + NUM_INSTANCES, i : i + NUM_INSTANCES]) for i in range(0, scores.shape[0], NUM_INSTANCES))
                # negative_score = torch.sum(scores) - positive_score
                # positive_score = positive_score / (count * count)
                # negative_score = negative_score / (count * count)
                scores = confusion_matrix(embeddings, embeddings) 
                plt.figure(figsize=(20, 20))
                plt.imshow(scores) 
                plt.colorbar() 
                plt.savefig(f"logs/{EXPERIMENT_NAME}/rgb_{epoch}.jpg")

            batch_size = cfg.SOLVER.IMS_PER_BATCH
            embeddings = torch.empty((len(val_loader) * batch_size, EMBEDDING_DIM))
            count = 0
            evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator.reset()
            with torch.no_grad():
                for n_iter, (img, depth, vid, target_cam, target_view) in enumerate(val_loader):
                    img = img.to(device)
                    depth = depth.to(device)
                    target = vid.to(device)
                    target_cam = target_cam.to(device)
                    target_view = target_view.to(device)
                    feat = model(img, depth, target, cam_label=target_cam, view_label=target_view )
                    feat = feat.cpu()
                    evaluator.update((feat, vid.cpu(), torch.tensor([0.0])))
                    # loss = loss_fn(score, feat, target, target_cam)
                    embeddings[n_iter * batch_size : n_iter * batch_size + img.shape[0]] = feat
                    count += img.shape[0]
                cmc, mAP, _, _, _, _, _ = evaluator.compute() 
                combined_mAP = mAP
                embeddings = embeddings[:count]
                # scores = F.cosine_similarity(embeddings.unsqueeze(0).repeat(count, 1, 1), embeddings.unsqueeze(1).repeat(1, count, 1))

                # print(f"computing the confusion matrix for the combined use case...")
                # scores = torch.zeros((count, count))
                # for i in tqdm(range(scores.shape[0])):
                #     for j in range(scores.shape[1]):
                #         scores[i][j] = embeddings[i] @ embeddings[j] / (torch.norm(embeddings[i]) * torch.norm(embeddings[j]))
                # positive_score = sum(torch.sum(scores[i : i + NUM_INSTANCES, i : i + NUM_INSTANCES]) for i in range(0, scores.shape[0], NUM_INSTANCES))
                # negative_score = torch.sum(scores) - positive_score
                # positive_score = positive_score / (count * count)
                # negative_score = negative_score / (count * count)
                scores = confusion_matrix(embeddings, embeddings)
                plt.figure(figsize=(20, 20))
                plt.imshow(scores) 
                plt.colorbar() 
                plt.savefig(f"logs/{EXPERIMENT_NAME}/rgb_depth_{epoch}.jpg")



                if WANDB:
                    wandb.log({
                        f"train_acc": acc_meter.avg,
                        f"train_loss": loss_meter.avg,
                        f"lr": scheduler._get_lr(epoch)[0],
                        f"epoch": epoch,
                        f"depth_confusion_matrix": wandb.Image(f"logs/{EXPERIMENT_NAME}/depth_{epoch}.jpg"),
                        f"rgb_confusion_matrix": wandb.Image(f"logs/{EXPERIMENT_NAME}/rgb_{epoch}.jpg"),
                        f"combined_confusion_matrix": wandb.Image(f"logs/{EXPERIMENT_NAME}/rgb_depth_{epoch}.jpg"),
                        f"depth_mAP": depth_mAP,
                        f"rgb_mAP": rgb_mAP,
                        f"combined_mAP": combined_mAP,
                    })
            torch.cuda.empty_cache()




                # for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                #     with torch.no_grad():
                #         img = img.to(device)
                #         camids = camids.to(device)
                #         target_view = target_view.to(device)
                #         feat = model(img, cam_label=camids, view_label=target_view)
                #         evaluator.update((feat, vid, camid))
                # cmc, mAP, _, _, _, _, _ = evaluator.compute()
                # logger.info("Validation Results - Epoch: {}".format(epoch))
                # logger.info("mAP: {:.1%}".format(mAP))
                # for r in [1, 5, 10]:
                #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                # torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


