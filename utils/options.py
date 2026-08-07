#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def str2bool(value):
    """Parse flexible command-line boolean values on Python 3.6+."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def add_bool_argument(group, name, default, help_text):
    """Add a boolean option that accepts either a bare flag or an explicit value."""
    group.add_argument(
        "--" + name,
        type=str2bool,
        nargs="?",
        const=True,
        default=default,
        metavar="BOOL",
        help=help_text,
    )


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=120, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--alr', type=float, default=0.1, help="attacker learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn','mlp','resnet_mnist','lstm', 'cnn_highacc'], help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=str, help='iid or qty or dir...')
    parser.add_argument('--alpha', type=float, default=0.5, help="level of non-iid in dirichlet_distribution_label_imbalance,used in Moon,etc.")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--attack', action='store_true', help='backdoor attack')
    parser.add_argument('--backdoor_baseline', type=str, default='standard',
                        choices=['standard', 'DBA', 'Neurotoxin'],
                        help='backdoor baseline: standard, DBA, or Neurotoxin')
    parser.add_argument('--num_attacker', type=int, default=1, help='number of attacker (default: 1)')
    parser.add_argument('--attack_start_round', type=int, default=4,
                        help='communication round when malicious clients begin attacking')    
    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=20,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip')
    parser.add_argument('--dp_sample', type=float, default=1, help='sample rate for moment account')

    parser.add_argument('--serial', action='store_true', help='partial serial running to save the gpu memory')
    parser.add_argument('--serial_bs', type=int, default=128, help='partial serial running batch size')
    parser.add_argument('--dim', type=int, default=28, help='noise dataset image size (same as training image)')
    parser.add_argument('--clipping_thresholds', type=float, default=1 / 3, help="clipping thresholds of update norm")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_samples', type=int, default=20000, help="number of samples of noise dataset")
    parser.add_argument('--attack_type', type=str, default='Collusion', help="Specific the attackers' behavior (Collusion, Input, Output)")
    parser.add_argument('--defense', type=str, help="Defense method")
    parser.add_argument('--log_distance', type=bool, default=False, help="output krum distance")
    parser.add_argument('--PDR', type=float, default=1, help="poison data rate")
    parser.add_argument('--neuro_mask_ratio', type=float, default=0.95,
                        help='fraction of small-gradient parameters kept by Neurotoxin')
    parser.add_argument('--neuro_aggregate_all_layer', type=int, default=0,
                        help='use one global Neurotoxin mask threshold when set to 1')
    parser.add_argument('--flame_noise', type=float, default=0.001, help='Flame noise')
    parser.add_argument('--save', type=str, default='save', help="dic to save results (ending without /)")
    parser.add_argument('--silhouette_threshold', type=float, default=0.5, help="Minimum silhouette score for clustering.")
    parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters for K-means in FLShield.")
    parser.add_argument('--num_validators', type=int, default=5, help="Number of validators in FLShield.")
    parser.add_argument('--loss_threshold', type=float, default=0.1, help="Loss threshold for selecting representative models.")
    parser.add_argument('--clip_value', type=float, default=1.0, help="Clipping value for bounding model updates.")
    parser.add_argument('--server_dataset', type=int, default=100, help="number of dataset in server")
    parser.add_argument('--plr_class', type=int, default=6, help="get PLRs of a specific class")
    parser.add_argument('--tau', type=float, default=0.8, help="threshold of LPA_ER")
    parser.add_argument('--thre_labels', type=float, default=2, help="threhold of labels per client in  quantity_based_non-iid_label_imbalance,used in FedAvg,etc.")
    parser.add_argument('--total_run', type=int, default=5, help="how many rounds totally run")
    parser.add_argument('--round_indicator', type=int, default=0, help="round number to start")
    parser.add_argument('--re_weight', action='store_true', help="use the dataset size to reweight the update")
    parser.add_argument('--vocabSize', type=int, default=0, help="vocabsize for sent140")
    parser.add_argument('--central_noise', type=float, default=0, help="std of the central noise")
    parser.add_argument('--random_drop', type=float, default=0, help="std of the central noise")

    # Probe Group defense: probe construction and suspicious-group scoring
    probe = parser.add_argument_group('Probe Group defense')
    probe.add_argument('--probe_target', type=int, default=1,
                       help='target class used by the probe trigger')
    probe.add_argument('--probe_batch_size', type=int, default=64,
                       help='batch size for each clean/probe evaluation batch')
    probe.add_argument('--probe_num_batches', type=int, default=2,
                       help='number of fixed test batches used for probing')
    probe.add_argument('--probe_patch_size', type=int, default=8,
                       help='side length of the square image probe patch')
    probe.add_argument('--probe_group_size', type=int, default=0,
                       help='clients per probe group; 0 selects the defense default')
    probe.add_argument('--probe_num_groups', type=int, default=0,
                       help='number of random probe groups; 0 selects the defense default')
    probe.add_argument('--probe_score_mode', type=str, default='excess_only',
                       choices=['excess_only', 'target_only', 'original', 'collusion'],
                       help='group anomaly score formulation')
    probe.add_argument('--probe_suspicious_fraction', type=float, default=0.25,
                       help='fraction of positive-excess candidate groups retained as suspicious')
    probe.add_argument('--probe_threshold_z', type=float, default=0.10,
                       help='MAD multiplier used for suspicious-group score thresholding')
    probe.add_argument('--probe_min_suspicious_groups', type=int, default=1,
                       help='minimum suspicious groups retained when candidates exist')
    add_bool_argument(probe, 'probe_require_positive_excess', True,
                      'require trigger gap to exceed clean gap before a group can be suspicious')
    probe.add_argument('--probe_excess_floor', type=float, default=0.0,
                       help='absolute minimum trigger-specific excess for candidate groups')
    probe.add_argument('--probe_excess_floor_z', type=float, default=0.0,
                       help='MAD multiplier for the adaptive trigger-excess floor')
    probe.add_argument('--probe_min_appearances', type=float, default=argparse.SUPPRESS,
                       help='appearances needed for full client-risk confidence; omitted uses the dynamic defense default')

    # Probe Group defense: attack-evidence gating
    add_bool_argument(probe, 'probe_use_gt_attack_gate', False,
                      'use known attacker presence as an experimental attack-evidence gate')
    add_bool_argument(probe, 'probe_gt_gate_requires_evidence', True,
                      'require unsupervised probe evidence even when the ground-truth gate is active')
    add_bool_argument(probe, 'probe_gate_memory_updates', True,
                      'allow positive attack evidence to update persistent risk memory')
    add_bool_argument(probe, 'probe_gate_hard_drop', True,
                      'allow positive attack evidence to authorize hard dropping')
    add_bool_argument(probe, 'probe_gate_soft_penalty', True,
                      'allow positive attack evidence to authorize soft penalties')
    probe.add_argument('--probe_attack_gate_min_excess', type=float, default=0.01,
                       help='minimum absolute excess required by the attack-evidence gate')
    probe.add_argument('--probe_attack_gate_z', type=float, default=2.0,
                       help='MAD multiplier for the attack-evidence excess threshold')
    probe.add_argument('--probe_attack_gate_min_groups', type=int, default=3,
                       help='minimum number of positive groups required by the attack gate')
    probe.add_argument('--probe_attack_gate_min_fraction', type=float, default=0.01,
                       help='minimum positive-group fraction required by the attack gate')
    add_bool_argument(probe, 'probe_no_evidence_use_fedavg', True,
                      'fall back to ordinary base weights when attack evidence is absent')

    # Probe Group defense: current and persistent client/pair risk
    probe.add_argument('--probe_pair_weight', type=float, default=0.70,
                       help='pair-pressure share of current-round client risk')
    probe.add_argument('--probe_risk_round_cap', type=float, default=0.60,
                       help='maximum current-round client risk contribution')
    probe.add_argument('--probe_risk_decay', type=float, default=0.85,
                       help='per-round decay applied to persistent individual risk')
    probe.add_argument('--probe_max_risk', type=float, default=argparse.SUPPRESS,
                       help='persistent risk cap; omitted preserves the defense context-specific defaults')
    probe.add_argument('--probe_pair_decay', type=float, default=0.92,
                       help='per-round decay applied to persistent pair risk')
    probe.add_argument('--probe_pair_max_risk', type=float, default=3.0,
                       help='maximum stored persistent risk for a client pair')
    probe.add_argument('--probe_pair_round_cap', type=float, default=0.80,
                       help='maximum pair evidence added in one round')
    probe.add_argument('--probe_pair_topk', type=int, default=3,
                       help='number of strongest pair relationships averaged per client')
    probe.add_argument('--probe_persistent_pair_mix', type=float, default=0.80,
                       help='pair-pressure share of combined persistent client risk')
    probe.add_argument('--probe_risk_grace', type=float, default=0.02,
                       help='persistent risk ignored before soft downweighting begins')
    probe.add_argument('--probe_tau', type=float, default=4.0,
                       help='exponential soft-downweighting strength')
    probe.add_argument('--probe_max_weight_drop', type=float, default=0.98,
                       help='maximum fractional soft weight reduction')

    # Probe Group defense: hard drop, catastrophic guard, and boundary rescue
    add_bool_argument(probe, 'probe_hard_drop', True,
                      'enable attack-gated hard rejection of high-risk clients')
    probe.add_argument('--probe_hard_drop_k', type=int, default=argparse.SUPPRESS,
                       help='number of clients to hard drop; omitted uses num_attacker')
    probe.add_argument('--probe_hard_drop_min_risk', type=float, default=0.0,
                       help='minimum persistent risk required for hard dropping')
    add_bool_argument(probe, 'probe_catastrophic_guard', False,
                      'veto hard dropping when suspicious evidence is diffuse or poorly localized')
    probe.add_argument('--probe_catastrophic_broad_fraction', type=float, default=0.18,
                       help='positive-group fraction considered broadly distributed evidence')
    probe.add_argument('--probe_catastrophic_saturation_fraction', type=float, default=0.25,
                       help='positive-group fraction that independently triggers the hard-drop veto')
    probe.add_argument('--probe_catastrophic_min_topk_overlap', type=float, default=0.50,
                       help='minimum overlap between current and persistent top-k risk sets')
    probe.add_argument('--probe_catastrophic_min_rank_agreement', type=float, default=0.25,
                       help='minimum current/persistent risk-rank correlation')
    probe.add_argument('--probe_catastrophic_min_margin_ratio', type=float, default=0.01,
                       help='minimum normalized risk margin at the hard-drop boundary')
    probe.add_argument('--probe_catastrophic_required_weak_signals', type=int, default=2,
                       help='weak localization signals required to veto broad-evidence hard dropping')
    add_bool_argument(probe, 'probe_boundary_rescue', True,
                      'rerank clients near a coherent hard-drop boundary using individual risk')
    probe.add_argument('--probe_boundary_rescue_max_positive_fraction', type=float, default=0.18,
                       help='maximum positive-group fraction for boundary rescue')
    probe.add_argument('--probe_boundary_rescue_min_topk_overlap', type=float, default=0.80,
                       help='minimum top-k agreement required for boundary rescue')
    probe.add_argument('--probe_boundary_rescue_min_rank_agreement', type=float, default=0.80,
                       help='minimum rank agreement required for boundary rescue')
    probe.add_argument('--probe_boundary_rescue_pool_extra', type=int, default=3,
                       help='extra clients considered around the hard-drop boundary')
    probe.add_argument('--probe_boundary_rescue_final_weight', type=float, default=0.40,
                       help='weight of final persistent risk in boundary reranking')
    probe.add_argument('--probe_boundary_rescue_persistent_individual_weight', type=float, default=0.40,
                       help='weight of persistent individual risk in boundary reranking')
    probe.add_argument('--probe_boundary_rescue_current_individual_weight', type=float, default=0.20,
                       help='weight of current individual risk in boundary reranking')

    # Probe Group defense: final aggregation and diagnostics
    add_bool_argument(probe, 'probe_use_median_clip', True,
                      'median-norm clip client updates before defended aggregation')
    probe.add_argument('--probe_blend_alpha', type=float, default=1.0,
                       help='blend weight on defended update versus the full FedAvg update')
    add_bool_argument(probe, 'probe_write_visuals', True,
                      'write Probe Group client/group diagnostic files and plots')
    probe.add_argument('--probe_visual_top_groups', type=int, default=60,
                       help='maximum high-scoring groups written to visual diagnostics JSON')

    args = parser.parse_args()
    return args
