import torch
import os
import numpy as np
from .utils import *
from .bypass_bn import *
import torch.nn.functional as F


def loop_one_epoch_jo(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type="train",
    logging_name=None,
    best_acc=0,
):
    """Run one epoch for training, testing, or evaluation."""

    # Tracking variables
    loss, total, correct = 0, 0, 0
    clean_total, clean_correct = 0, 0
    noise_total, noise_correct = 0, 0
    noise_acc, clean_acc = 0, 0

    if "train" in loop_type:
        net.train()
        results = np.zeros((len(dataloader.dataset), dataloader.dataset.num_classes), dtype=np.float32)

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, noise_masks, soft_targets, indexs = [x.to(device) for x in batch]

            opt_name = type(optimizer).__name__

            # --- SGD case ---
            if opt_name == "SGD":
                outputs = net(inputs)
                if loop_type != 'retrain':
                    probs, first_loss = criterion(outputs, soft_targets)
                    results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
                else:
                    first_loss = criterion(outputs, soft_targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # metrics after update
                outputs_updated = net(inputs)
                if loop_type != 'retrain':
                    probs, first_loss = criterion(outputs_updated, soft_targets)
                else:
                    first_loss = criterion(outputs_updated, soft_targets)

            # --- SAM-like optimizer case ---
            else:
                # freeze BN on the ascent step and only update stats once
                disable_running_stats(net)
                outputs = net(inputs)

                optimizer.zero_grad()
                if loop_type != 'retrain':
                    probs, first_loss = criterion(outputs, soft_targets)
                    results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
                else:
                    first_loss = criterion(outputs, soft_targets)
                
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                enable_running_stats(net)
                if loop_type != 'retrain':
                    _, second_loss = criterion(net(inputs), soft_targets)
                    second_loss.backward()
                else:
                    criterion(net(inputs), soft_targets).backward()

                optimizer.second_step(zero_grad=True)

                # compute metrics on final weights
                outputs_updated = net(inputs)
                if loop_type != 'retrain':
                    probs, first_loss = criterion(outputs_updated, soft_targets)
                else:
                    first_loss = criterion(outputs_updated, soft_targets)

            # --- Evaluation within training ---
            with torch.no_grad():
                loss += float(first_loss.item())
                loss_mean = loss / (batch_idx + 1)

                _, predicted = outputs_updated.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total

                if len(batch) == 2:  # No noise masks
                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")
                else:  # With noise masks
                    noise_total += noise_masks.sum().item()
                    noise_correct += predicted.eq(targets).mul(noise_masks).sum().item()
                    noise_acc = 100. * noise_correct / (noise_total + 1e-6)

                    clean_total += targets.size(0) - noise_masks.sum().item()
                    clean_correct += predicted.eq(targets).mul(~noise_masks).sum().item()
                    clean_acc = 100. * clean_correct / (clean_total + 1e-6)

                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total}) | "
                                     f"Noise: {noise_acc:.3f}% ({noise_correct}/{noise_total}) | "
                                     f"Clean: {clean_acc:.3f}% ({clean_correct}/{clean_total})")
        if loop_type != 'retrain':
            dataloader.dataset.label_update(results)

        # Save noise/clean accuracies
        logging_dict[f"Train/noise_acc"] = noise_acc
        logging_dict[f"Train/clean_acc"] = clean_acc
        logging_dict[f"Train/gap_clean_noise_acc"] = clean_acc - noise_acc

    # --- Testing phase ---
    elif loop_type == "test":
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

            # Save best checkpoint
            if acc > best_acc:
                print("Saving best checkpoint ...")
                state = {
                    "net": net.state_dict(),
                    "acc": acc,
                    "loss": loss,
                    "epoch": epoch,
                }
                save_path = os.path.join("checkpoint", logging_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "ckpt_best.pth"))
                best_acc = acc

            logging_dict[f"{loop_type.title()}/best_acc"] = best_acc

        logging_dict[f"{loop_type.title()}/gen_gap"] = logging_dict["Train/acc"] - acc

    # --- Resume from checkpoint ---
    else:
        print("==> Resuming from best checkpoint..")
        save_path = os.path.join("checkpoint", logging_name)
        checkpoint = torch.load(os.path.join(save_path, "ckpt_best.pth"))
        net.load_state_dict(checkpoint["net"])
        net.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

    # Final logging
    if loop_type == 'retrain':
        logging_dict[f"Train/loss"] = loss_mean
        logging_dict[f"Train/acc"] = acc
    else:
        logging_dict[f"{loop_type.title()}/loss"] = loss_mean
        logging_dict[f"{loop_type.title()}/acc"] = acc

    if loop_type == "test":
        return best_acc, acc



def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type="train",
    logging_name=None,
    best_acc=0,
):
    """Run one epoch for training, testing, or evaluation."""

    # Tracking variables
    loss, total, correct = 0, 0, 0
    clean_total, clean_correct = 0, 0
    noise_total, noise_correct = 0, 0
    noise_acc, clean_acc = 0, 0
    noise_B_cosine_score, norm_grad_B, norm_noise_grad = [], [], []

    if loop_type == "train":
        net.train()

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, noise_masks = [x.to(device) for x in batch]
            noise_inputs, noise_targets = inputs[noise_masks == 1], targets[noise_masks == 1]

            opt_name = type(optimizer).__name__

            # --- SGD case ---
            if opt_name == "SGD":
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # --- SAM-like optimizer case ---
            else:
                # freeze BN statistics while climbing
                disable_running_stats(net)
                outputs = net(inputs)

                # Extra noise gradient logging every 8 batches
                if (batch_idx + 1) % 8 == 0:
                    noise_outputs = outputs[noise_masks]
                    batch_size = outputs.shape[0]
                    num_noise = noise_inputs.shape[0]
                    noise_loss = criterion(noise_outputs, noise_targets) * (num_noise / batch_size)
                    noise_loss.backward(retain_graph=True)
                    noise_grads = get_gradients(optimizer)

                optimizer.zero_grad()
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                enable_running_stats(net)
                criterion(net(inputs), targets).backward()

                # Compare gradients between noise and B group
                if (batch_idx + 1) % 8 == 0:
                    B_grads, _ = get_grads_and_masks_at_group(optimizer)
                    for grad1, grad2 in zip(B_grads, noise_grads):
                        dot = torch.sum(grad1 * grad2)
                        norm1, norm2 = torch.norm(grad1), torch.norm(grad2)
                        cosine_sim = dot / (norm1 * norm2 + 1e-18)

                        norm_grad_B.append(norm1.item())
                        norm_noise_grad.append(norm2.item())
                        noise_B_cosine_score.append(cosine_sim.item())

                # Log gradient statistics at the end of each epoch
                if (batch_idx + 1) % len(dataloader) == 0:
                    logging_dict.update(get_checkpoint(optimizer))
                    logging_dict.update({
                        "prop/noise_B_cosine_score": np.mean(noise_B_cosine_score),
                        "prop/norm_grad_B": np.mean(norm_grad_B),
                        "prop/norm_noise_grad": np.mean(norm_noise_grad),
                    })

                optimizer.second_step(zero_grad=True)

                # recompute outputs after both steps for logging
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

            # --- Evaluation within training ---
            with torch.no_grad():
                loss += float(first_loss.item())
                loss_mean = loss / (batch_idx + 1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total

                if noise_masks is None:  # No noise masks
                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")
                else:  # With noise masks
                    noise_total += noise_masks.sum().item()
                    noise_correct += predicted.eq(targets).mul(noise_masks).sum().item()
                    noise_acc = 100. * noise_correct / (noise_total + 1e-6)

                    clean_total += targets.size(0) - noise_masks.sum().item()
                    clean_correct += predicted.eq(targets).mul(~noise_masks).sum().item()
                    clean_acc = 100. * clean_correct / (clean_total + 1e-6)

                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total}) | "
                                     f"Noise: {noise_acc:.3f}% ({noise_correct}/{noise_total}) | "
                                     f"Clean: {clean_acc:.3f}% ({clean_correct}/{clean_total})")

        # Save noise/clean accuracies
        logging_dict[f"{loop_type.title()}/noise_acc"] = noise_acc
        logging_dict[f"{loop_type.title()}/clean_acc"] = clean_acc
        logging_dict[f"{loop_type.title()}/gap_clean_noise_acc"] = clean_acc - noise_acc

    # --- Testing phase ---
    elif loop_type == "test":
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

            # Save best checkpoint
            if acc > best_acc:
                print("Saving best checkpoint ...")
                state = {
                    "net": net.state_dict(),
                    "acc": acc,
                    "loss": loss,
                    "epoch": epoch,
                }
                save_path = os.path.join("checkpoint", logging_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "ckpt_best.pth"))
                best_acc = acc

            logging_dict[f"{loop_type.title()}/best_acc"] = best_acc

        if "Train/acc" in logging_dict:
            logging_dict["Test/gen_gap"] = logging_dict["Train/acc"] - acc

    # --- Resume from checkpoint ---
    else:
        print("==> Resuming from best checkpoint..")
        save_path = os.path.join("checkpoint", logging_name)
        checkpoint = torch.load(os.path.join(save_path, "ckpt_best.pth"))
        net.load_state_dict(checkpoint["net"])
        net.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

    # Final logging
    if loop_type == 'retrain':
        logging_dict[f"Train/loss"] = loss_mean
        logging_dict[f"Train/acc"] = acc
    else:
        logging_dict[f"{loop_type.title()}/loss"] = loss_mean
        logging_dict[f"{loop_type.title()}/acc"] = acc

    if loop_type == "test":
        return best_acc, acc


def remember_rate_schedule(epoch, total_epochs, forget_rate, num_gradual=10, exponent=1.0):
    """
    Co-teaching schedule:
      remember_rate = 1 - forget_rate_t
    where forget_rate_t increases gradually to forget_rate.
    """
    if num_gradual <= 0:
        fr_t = forget_rate
    else:
        t = min(epoch, num_gradual)
        fr_t = forget_rate * (t / num_gradual) ** exponent
    remember_rate = 1.0 - fr_t
    return max(0.0, min(1.0, remember_rate))


def coteaching_select_indices(loss_vec, remember_rate):
    """
    loss_vec: [B] per-sample loss
    return: indices of smallest losses (size ~= remember_rate * B)
    """
    B = loss_vec.numel()
    k = int(remember_rate * B)
    k = max(1, min(B, k))
    _, idx_sorted = torch.sort(loss_vec.detach())  # ascending
    return idx_sorted[:k]


def loop_one_epoch_co_teaching(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type="train",
    logging_name=None,
    best_acc=0,
    # --- Co-teaching args ---
    coteaching=False,
    criterion_vec=None,
    forget_rate=0.2,
    num_gradual=10,
    exponent=1.0,
    total_epochs=None,
):
    """
    If coteaching=True:
      - net must be (net1, net2)
      - optimizer must be (opt1, opt2)
      - criterion_vec must be CrossEntropyLoss(reduction="none")
    """

    loss, total, correct = 0, 0, 0
    clean_total, clean_correct = 0, 0
    noise_total, noise_correct = 0, 0
    noise_acc, clean_acc = 0, 0
    loss_mean, acc = 0.0, 0.0

    if loop_type == "train":
        if coteaching:
            assert isinstance(net, (tuple, list)) and len(net) == 2
            assert isinstance(optimizer, (tuple, list)) and len(optimizer) == 2
            assert criterion_vec is not None, "Need per-sample loss: criterion_vec with reduction='none'"
            net1, net2 = net
            opt1, opt2 = optimizer
            net1.train(); net2.train()

            remember_rate = remember_rate_schedule(
                epoch=epoch,
                total_epochs=total_epochs if total_epochs is not None else epoch,
                forget_rate=forget_rate,
                num_gradual=num_gradual,
                exponent=exponent,
            )
            logging_dict["CoTeach/remember_rate"] = remember_rate

            for batch_idx, batch in enumerate(dataloader):
                # supports (inputs, targets) OR (inputs, targets, noise_masks)
                if len(batch) == 4:
                    inputs, targets, noise_masks, indices = batch
                elif len(batch) == 3:
                    inputs, targets, noise_masks = batch
                    indices = None
                else:
                    inputs, targets = batch
                    noise_masks = None
                    indices = None

                inputs = inputs.to(device)
                targets = targets.to(device)
                if noise_masks is not None:
                    noise_masks = noise_masks.to(device)

                # forward both nets
                out1 = net1(inputs)
                out2 = net2(inputs)

                # per-sample losses for selection
                lvec1 = criterion_vec(out1, targets)  # [B]
                lvec2 = criterion_vec(out2, targets)  # [B]

                idx1_small = coteaching_select_indices(lvec1, remember_rate)
                idx2_small = coteaching_select_indices(lvec2, remember_rate)

                # exchange: net1 learns from net2's selected; net2 learns from net1's selected
                idx_for_net1 = idx2_small
                idx_for_net2 = idx1_small

                # --- update net1 ---
                opt_name1 = type(opt1).__name__
                if opt_name1 == "SGD":
                    loss1 = criterion(out1[idx_for_net1], targets[idx_for_net1])
                    opt1.zero_grad()
                    loss1.backward()
                    opt1.step()
                else:
                    # running stats should be frozen during the ascent step; we only
                    # update them once with the *clean* weights.  you had the
                    # calls reversed previously which meant BN statistics were
                    # being collected on the perturbed parameters. that noise
                    # corrupts the stored mean/var and is the usual cause of
                    # good training numbers + exploding test metrics.
                    disable_running_stats(net1)
                    loss1 = criterion(out1[idx_for_net1], targets[idx_for_net1])
                    opt1.zero_grad()
                    loss1.backward()
                    opt1.first_step(zero_grad=True)

                    enable_running_stats(net1)
                    out1b = net1(inputs)
                    loss1b = criterion(out1b[idx_for_net1], targets[idx_for_net1])
                    loss1b.backward()
                    opt1.second_step(zero_grad=True)

                # --- update net2 ---
                opt_name2 = type(opt2).__name__
                if opt_name2 == "SGD":
                    loss2 = criterion(out2[idx_for_net2], targets[idx_for_net2])
                    opt2.zero_grad()
                    loss2.backward()
                    opt2.step()
                else:
                    disable_running_stats(net2)
                    loss2 = criterion(out2[idx_for_net2], targets[idx_for_net2])
                    opt2.zero_grad()
                    loss2.backward()
                    opt2.first_step(zero_grad=True)

                    enable_running_stats(net2)
                    out2b = net2(inputs)
                    loss2b = criterion(out2b[idx_for_net2], targets[idx_for_net2])
                    loss2b.backward()
                    opt2.second_step(zero_grad=True)

                # --- logging (use net1 predictions for acc; or ensemble if you want) ---
                with torch.no_grad():
                    # re‑evaluate with the **updated** parameters, not the stale
                    # `out1` computed before either of the SAM steps. this is the
                    # model that will actually be used at test time.
                    updated = net1(inputs)
                    batch_loss = 0.5 * (
                        criterion(updated[idx_for_net1], targets[idx_for_net1]).item()
                        + criterion(net2(inputs)[idx_for_net2], targets[idx_for_net2]).item()
                    )
                    loss += float(batch_loss)
                    loss_mean = loss / (batch_idx + 1)

                    _, pred1 = updated.max(1)
                    total += targets.size(0)
                    correct += pred1.eq(targets).sum().item()
                    acc = 100.0 * correct / total

                    if noise_masks is not None:
                        noise_total += noise_masks.sum().item()
                        noise_correct += pred1.eq(targets).mul(noise_masks.bool()).sum().item()
                        noise_acc = 100.0 * noise_correct / (noise_total + 1e-6)

                        clean_total += targets.size(0) - noise_masks.sum().item()
                        clean_correct += pred1.eq(targets).mul((~noise_masks.bool())).sum().item()
                        clean_acc = 100.0 * clean_correct / (clean_total + 1e-6)

                    if batch_idx % max(1, (len(dataloader) // 10)) == 0 or (batch_idx + 1) == len(dataloader):
                        msg = f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})"
                        if noise_masks is not None:
                            msg += f" | Noise: {noise_acc:.3f}% ({noise_correct}/{noise_total}) | Clean: {clean_acc:.3f}% ({clean_correct}/{clean_total})"
                        progress_bar(batch_idx, len(dataloader), msg)

            # final epoch logs
            logging_dict["Train/loss"] = loss_mean
            logging_dict["Train/acc"] = acc
            if noise_masks is not None:
                logging_dict["Train/noise_acc"] = noise_acc
                logging_dict["Train/clean_acc"] = clean_acc
                logging_dict["Train/gap_clean_noise_acc"] = clean_acc - noise_acc

            return  # training doesn't return (best_acc, acc)

        # -------------------------
        # your ORIGINAL single-model path (kept minimal)
        # -------------------------
        else:
            net.train()
            for batch_idx, batch in enumerate(dataloader):
                inputs, targets, noise_masks = [x.to(device) for x in batch]
                opt_name = type(optimizer).__name__

                if opt_name == "SGD":
                    outputs = net(inputs)
                    first_loss = criterion(outputs, targets)
                    first_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # metrics will be computed on the updated weights below
                    outputs_updated = net(inputs)
                    first_loss = criterion(outputs_updated, targets)
                else:
                    # freeze batchnorm while climbing the loss landscape
                    disable_running_stats(net)
                    outputs = net(inputs)
                    optimizer.zero_grad()
                    first_loss = criterion(outputs, targets)
                    first_loss.backward()
                    optimizer.first_step(zero_grad=True)

                    enable_running_stats(net)
                    outputs2 = net(inputs)
                    loss2 = criterion(outputs2, targets)
                    loss2.backward()
                    optimizer.second_step(zero_grad=True)

                    # collect metrics using final parameters
                    outputs_updated = net(inputs)
                    first_loss = criterion(outputs_updated, targets)

                with torch.no_grad():
                    loss += float(first_loss.item())
                    loss_mean = loss / (batch_idx + 1)
                    _, predicted = outputs_updated.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = 100. * correct / total

            logging_dict["Train/loss"] = loss_mean
            logging_dict["Train/acc"] = acc
            return

    elif loop_type == "test":
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % max(1, (len(dataloader) // 10)) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

            if acc > best_acc:
                print("Saving best checkpoint ...")
                state = {"net": net.state_dict(), "acc": acc, "loss": loss, "epoch": epoch}
                save_path = os.path.join("checkpoint", logging_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "ckpt_best.pth"))
                best_acc = acc

            logging_dict["Test/best_acc"] = best_acc
            logging_dict["Test/loss"] = loss_mean
            logging_dict["Test/acc"] = acc

        # optional: only if Train/acc exists (after training)
        if "Train/acc" in logging_dict:
            logging_dict["Test/gen_gap"] = logging_dict["Train/acc"] - acc

        return best_acc, acc


def sharpen(p, T=0.5):
    p = p ** (1.0 / T)
    return p / (p.sum(dim=1, keepdim=True) + 1e-12)

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

def mixup(x, y, alpha=4.0):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix, lam

@torch.no_grad()
def eval_loss_per_sample(net, dataloader, device, criterion_vec, n_samples):
    """
    Returns:
      losses: numpy array [n_samples]
    Requires dataloader to return indices.
    """
    net.eval()
    losses = np.zeros(n_samples, dtype=np.float32)

    for batch in dataloader:
        # (inputs, targets, noise_masks, indices) OR (inputs, targets, indices)
        if len(batch) == 4:
            inputs, targets, noise_masks, indices = batch
        elif len(batch) == 3:
            inputs, targets, noise_masks = batch
            indices = None
        else:
            inputs, targets = batch
            noise_masks = None
            indices = None


        inputs, targets = inputs.to(device), targets.to(device)
        indices = indices.cpu().numpy()

        logits = net(inputs)
        lvec = criterion_vec(logits, targets).detach().cpu().numpy()
        losses[indices] = lvec
    return losses

def fit_gmm_two_components(losses):
    """
    Practical note:
    DivideMix uses a 2-component Gaussian Mixture Model on *normalized losses*.
    Here is a simple, dependency-free EM for 2 Gaussians (diagonal, 1D).
    """
    x = losses.astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)

    # init
    mu1, mu2 = np.quantile(x, 0.3), np.quantile(x, 0.7)
    var1, var2 = np.var(x) + 1e-6, np.var(x) + 1e-6
    pi1, pi2 = 0.5, 0.5

    for _ in range(25):
        # E-step
        p1 = pi1 * np.exp(-0.5 * (x - mu1) ** 2 / var1) / np.sqrt(var1 + 1e-12)
        p2 = pi2 * np.exp(-0.5 * (x - mu2) ** 2 / var2) / np.sqrt(var2 + 1e-12)
        gamma1 = p1 / (p1 + p2 + 1e-12)
        gamma2 = 1.0 - gamma1

        # M-step
        N1, N2 = gamma1.sum() + 1e-12, gamma2.sum() + 1e-12
        mu1 = (gamma1 * x).sum() / N1
        mu2 = (gamma2 * x).sum() / N2
        var1 = (gamma1 * (x - mu1) ** 2).sum() / N1 + 1e-6
        var2 = (gamma2 * (x - mu2) ** 2).sum() / N2 + 1e-6
        pi1 = N1 / (N1 + N2)
        pi2 = 1.0 - pi1

    # clean component = smaller mean
    if mu1 < mu2:
        p_clean = gamma1
    else:
        p_clean = gamma2

    return p_clean.astype(np.float32)  # probability of being clean

def loop_one_epoch_warmup(dataloader, net, optimizer, device, criterion, logging_dict, epoch, logging_name, tag="net"):
    net.train()
    loss_sum, total, correct = 0.0, 0, 0

    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 4:
            inputs, targets, noise_masks, indices = batch
        elif len(batch) == 3:
            inputs, targets, noise_masks = batch
            indices = None
        else:
            inputs, targets = batch
            noise_masks = None
            indices = None


        inputs, targets = inputs.to(device), targets.to(device)
        
        opt_name = type(optimizer).__name__
        if opt_name == "SGD":
            outputs = net(inputs)
            first_loss = criterion(outputs, targets)
            first_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            outputs = net(inputs)
            first_loss = criterion(outputs, targets)
        else:
            disable_running_stats(net)
            outputs = net(inputs)
            optimizer.zero_grad()
            first_loss = criterion(outputs, targets)
            first_loss.backward()
            optimizer.first_step(zero_grad=True)

            enable_running_stats(net)
            second_loss = criterion(net(inputs), targets)
            second_loss.backward()
            optimizer.second_step(zero_grad=True)

            outputs = net(inputs)
            first_loss = criterion(outputs, targets)

        with torch.no_grad():
            loss_sum += first_loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        if batch_idx % max(1, len(dataloader)//10) == 0 or (batch_idx+1)==len(dataloader):
            progress_bar(batch_idx, len(dataloader),
                         f"[Warmup {tag}] Loss: {loss_sum/(batch_idx+1):.3f} | Acc: {100.0*correct/total:.2f}%")

    logging_dict[f"Warmup/{tag}_loss"] = loss_sum / max(1, len(dataloader))
    logging_dict[f"Warmup/{tag}_acc"] = 100.0 * correct / max(1, total)


def dividemix_step(
    inputs,
    targets,
    indices_np,
    net,
    net_other,
    p_clean_other,
    device,
    num_classes,
    p_threshold=0.5,
    T=0.5,
    alpha=4.0,
    lambda_u=25.0,
):
    """
    One DivideMix training step (forward + loss computation)

    Returns:
        loss, loss_x, loss_u, stats_dict
    """

    # -----------------------
    # 1) Split labeled / unlabeled
    # -----------------------
    p_clean = torch.from_numpy(p_clean_other[indices_np]).to(device)  # [B]
    is_labeled = (p_clean >= p_threshold)
    is_unlabeled = ~is_labeled

    if is_labeled.sum() == 0:
        return None, None, None, {"skip": True}

    x_l = inputs[is_labeled]
    y_l = targets[is_labeled]

    x_u = inputs[is_unlabeled]

    # -----------------------
    # 2) Label refinement (labeled)
    # -----------------------
    with torch.no_grad():
        p = torch.softmax(net(x_l), dim=1)
        y_one = one_hot(y_l, num_classes)

        w = p_clean[is_labeled].unsqueeze(1)  # [Bl,1]
        y_refine = w * y_one + (1 - w) * p
        y_refine = y_refine / (y_refine.sum(dim=1, keepdim=True) + 1e-12)

    # -----------------------
    # 3) Pseudo-label (unlabeled)
    # -----------------------
    if x_u.numel() > 0:
        with torch.no_grad():
            pu1 = torch.softmax(net(x_u), dim=1)
            pu2 = torch.softmax(net_other(x_u), dim=1)
            pu = 0.5 * (pu1 + pu2)
            y_u = sharpen(pu, T=T)
    else:
        y_u = None

    # -----------------------
    # 4) MixUp
    # -----------------------
    if x_u.numel() > 0:
        x_all = torch.cat([x_l, x_u], dim=0)
        y_all = torch.cat([y_refine, y_u], dim=0)
    else:
        x_all = x_l
        y_all = y_refine

    x_mix, y_mix, _ = mixup(x_all, y_all, alpha=alpha)

    # split back
    Bl = x_l.size(0)
    x_mix_l, y_mix_l = x_mix[:Bl], y_mix[:Bl]

    if x_u.numel() > 0:
        x_mix_u, y_mix_u = x_mix[Bl:], y_mix[Bl:]
    else:
        x_mix_u, y_mix_u = None, None

    # -----------------------
    # 5) Loss computation
    # -----------------------
    logits = net(x_mix)

    logits_l = logits[:Bl]
    loss_x = -(y_mix_l * torch.log_softmax(logits_l, dim=1)).sum(dim=1).mean()

    if x_u is not None and x_u.numel() > 0:
        logits_u = logits[Bl:]
        pu = torch.softmax(logits_u, dim=1)
        loss_u = torch.mean((pu - y_mix_u) ** 2)
    else:
        loss_u = torch.tensor(0.0, device=device)

    loss = loss_x + lambda_u * loss_u

    # -----------------------
    # 6) Stats
    # -----------------------
    stats = {
        "labeled_bs": int(is_labeled.sum().item()),
        "unlabeled_bs": int(is_unlabeled.sum().item()),
    }

    return loss, loss_x, loss_u


def train_dividemix_epoch(
    dataloader,
    net,
    net_other,
    optimizer,
    device,
    num_classes,
    p_clean_other,      # numpy [N]
    p_threshold=0.5,
    lambda_u=25.0,
    T=0.5,
    alpha=4.0,
    logging_dict=None,
    epoch=0,
    logging_name=None,
    tag="net",
):
    net.train()
    net_other.eval()

    loss_sum, loss_x_sum, loss_u_sum = 0.0, 0.0, 0.0
    total, correct = 0, 0

    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 4:
            inputs, targets, noise_masks, indices = batch
        elif len(batch) == 3:
            inputs, targets, noise_masks = batch
            indices = None
        else:
            inputs, targets = batch
            noise_masks = None
            indices = None

        inputs = inputs.to(device)
        targets = targets.to(device)
        indices_np = indices.cpu().numpy()

        opt_name = type(optimizer).__name__
        if opt_name == "SGD":
            first_loss, first_loss_x, first_loss_u = dividemix_step(inputs, targets, indices_np, net, net_other, p_clean_other, device, num_classes, p_threshold, T, alpha, lambda_u)

            first_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            outputs = net(inputs)
            _, pred = outputs.max(1)
        else:
            disable_running_stats(net)
            optimizer.zero_grad()
            first_loss, first_loss_x, first_loss_u = dividemix_step(inputs, targets, indices_np, net, net_other, p_clean_other, device, num_classes, p_threshold, T, alpha, lambda_u)
            first_loss.backward()
            optimizer.first_step(zero_grad=True)

            enable_running_stats(net)
            second_loss, _, _ = dividemix_step(inputs, targets, indices_np, net, net_other, p_clean_other, device, num_classes, p_threshold, T, alpha, lambda_u)
            second_loss.backward()
            optimizer.second_step(zero_grad=True)

            outputs = net(inputs)
            _, pred = outputs.max(1)

        # ---- Stats ----
        with torch.no_grad():
            # make sure we evaluate with the final parameters
            if opt_name != "SGD":
                first_loss, first_loss_x, first_loss_u = dividemix_step(
                    inputs, targets, indices_np, net, net_other, p_clean_other,
                    device, num_classes, p_threshold, T, alpha, lambda_u,
                )
            loss_sum += first_loss.item()
            loss_x_sum += first_loss_x.item()
            loss_u_sum += first_loss_u.item()

            # report accuracy on original inputs using hard labels (net predictions)
            logits_orig = net(inputs)
            _, pred = logits_orig.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        if batch_idx % max(1, len(dataloader)//10) == 0 or (batch_idx+1)==len(dataloader):
            progress_bar(batch_idx, len(dataloader),
                         f"[DivideMix {tag}] Lx: {loss_x_sum/(batch_idx+1):.3f} | "
                         f"Lu: {loss_u_sum/(batch_idx+1):.3f} | "
                         f"Acc: {100.0*correct/total:.2f}%")

    if logging_dict is not None:
        denom = max(1, len(dataloader))
        logging_dict[f"DivideMix/{tag}_loss"] = loss_sum / denom
        logging_dict[f"DivideMix/{tag}_loss_x"] = loss_x_sum / denom
        logging_dict[f"DivideMix/{tag}_loss_u"] = loss_u_sum / denom
        logging_dict[f"DivideMix/{tag}_acc"] = 100.0 * correct / max(1, total)
