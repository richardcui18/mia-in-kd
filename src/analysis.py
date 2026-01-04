import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import sys
import json
from analysis_options import Options
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import visualization


def read_tables_from_multiple_files(input_path_teacher, input_path_student):
    all_output_teacher, optimal_thresholds_teacher = visualization.read_tables_from_file_one_model(input_path_teacher)
    all_output_student, optimal_thresholds_student = visualization.read_tables_from_file_one_model(input_path_student)

    all_output = []
    for i, ex in enumerate(all_output_teacher):
        pred_stud_label_ex = {}
        pred_stud_ex = {}
        
        # Loop through each metric
        for metric in ex['pred'].keys():
            if all_output_student[i]['input'] == ex['input']:
                pred_stud_label_ex[metric] = int(all_output_student[i]['predicted_label'][metric])
                pred_stud_ex[metric] = float(all_output_student[i]['pred'][metric])
            else:
                print("ERROR: Input text mismatch")
        
        ex_teacher_student = {}
        ex_teacher_student['input'] = ex['input']
        ex_teacher_student['label'] = ex['label']
        ex_teacher_student['pred_teacher'] = ex['pred']
        ex_teacher_student['predicted_label_teacher'] = ex['predicted_label']
        ex_teacher_student['pred_student'] = pred_stud_ex
        ex_teacher_student['predicted_label_student'] = pred_stud_label_ex

        all_output.append(ex_teacher_student)

    return all_output, optimal_thresholds_teacher, optimal_thresholds_student



def calculate_metrics(all_output):
    true_labels = []
    
    # Prediction label containers
    teacher_preds = {k: [] for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']}
    student_preds = {k: [] for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']}
    
    # Score containers for AUC
    teacher_scores = {k: [] for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']}
    student_scores = {k: [] for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']}

    # Extract data
    for ex in all_output:
        true_labels.append(int(ex['label']))
        for metric in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']:
            teacher_preds[metric].append(int(ex['predicted_label_teacher'][metric]))
            student_preds[metric].append(int(ex['predicted_label_student'][metric]))
            teacher_scores[metric].append(float(ex['pred_teacher'][metric]))
            student_scores[metric].append(float(ex['pred_student'][metric]))

    # Accuracy & SE
    def get_accuracy_and_se(true, pred):
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tpr + tnr) / 2
        se_tpr = np.sqrt(tpr * (1 - tpr) / (tp + fn)) if (tp + fn) > 0 else 0
        se_tnr = np.sqrt(tnr * (1 - tnr) / (tn + fp)) if (tn + fp) > 0 else 0
        se = 0.5 * np.sqrt(se_tpr**2 + se_tnr**2)
        return acc, se

    # Compute Accuracy, SE
    teacher_accuracy, student_accuracy = [], []
    teacher_se, student_se = [], []

    for metric in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']:
        acc_t, se_t = get_accuracy_and_se(true_labels, teacher_preds[metric])
        acc_s, se_s = get_accuracy_and_se(true_labels, student_preds[metric])
        
        teacher_accuracy.append(acc_t)
        student_accuracy.append(acc_s)
        teacher_se.append(se_t)
        student_se.append(se_s)
        
    # AUCs
    try:
        teacher_auc = {k: roc_auc_score(true_labels, teacher_scores[k]) for k in teacher_scores}
        student_auc = {k: roc_auc_score(true_labels, student_scores[k]) for k in student_scores}

        # Optional: ROC Curve + Balanced Accuracy for `recall`
        fpr, tpr, _ = roc_curve(true_labels, teacher_scores['recall'])
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        print(f"teacher recall AUC {teacher_auc['recall']}")
        print(f"teacher recall accuracy {acc}")
    except ValueError as e:
        print("Warning: Could not compute AUC due to insufficient class separation:", str(e))
        teacher_auc = student_auc = {k: 0.0 for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']}

    prediction_summary = [
        true_labels,
        *[teacher_preds[k] for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']],
        *[student_preds[k] for k in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']],
    ]

    return teacher_accuracy, student_accuracy, teacher_se, student_se, prediction_summary



def plot_distribution_matrix(all_output, metric, teacher_model_name, student_model_name):
    matrix_distribution = {
        'teacher_1': {'student_1': 0, 'student_0': 0},
        'teacher_0': {'student_1': 0, 'student_0': 0}
    }

    for ex in all_output:
        label = int(ex['label'])
        pred_teacher = int(ex['predicted_label_teacher'][metric])
        pred_student = int(ex['predicted_label_student'][metric])

        if pred_teacher == 1 and pred_student == 1:
            matrix_distribution['teacher_1']['student_1'] += 1
        elif pred_teacher == 0 and pred_student == 1:
            matrix_distribution['teacher_0']['student_1'] += 1
        elif pred_teacher == 1 and pred_student == 0:
            matrix_distribution['teacher_1']['student_0'] += 1
        else:
            matrix_distribution['teacher_0']['student_0'] += 1

    total = len(all_output)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#fff5eb", "#fdbe85", "#fd8d3c", "#d94701"]
    cmap = LinearSegmentedColormap.from_list("academic_oranges", colors)

    matrix = np.array([
        [matrix_distribution['teacher_1']['student_1']/total, matrix_distribution['teacher_1']['student_0']/total],
        [matrix_distribution['teacher_0']['student_1']/total, matrix_distribution['teacher_0']['student_0']/total]
    ])

    ax = sns.heatmap(matrix, annot=False, cmap=cmap, linewidths=2, 
                     linecolor='white', cbar=False, square=True)

    ax.set_xticklabels(['Member', 'Non-member'], fontsize=14, fontweight='bold')
    ax.set_yticklabels(['Member', 'Non-member'], fontsize=14, fontweight='bold')

    ax.set_xlabel('MIA Result on Student Model', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('MIA Result on Teacher Model', fontsize=16, fontweight='bold', labelpad=15)

    cell_texts = [
        [f"{matrix[0,0]:.3f}", 
         f"{matrix[0,1]:.3f}"],
        [f"{matrix[1,0]:.3f}", 
         f"{matrix[1,1]:.3f}"]
    ]

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, cell_texts[i][j], 
                    ha="center", va="center", fontsize=16,
                    color="black" if matrix[i, j] < 0.25 else "white")

    plt.suptitle(f'Prediction Class Distribution Comparison ({metric.capitalize()})', 
                 fontsize=20, fontweight='bold', y=0.975)
    plt.title(f'{teacher_model_name} / {student_model_name}', 
              fontsize=16, pad=15)

    fig.text(0.5, 0.01, 
             "Each cell shows the proportion of training data in each category",
             ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)

    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)
    plt.savefig(os.path.join('../fig/single_student', teacher_model_name, student_model_name, 
                           f'{metric}_distribution_comparison_matrix.png'), 
                dpi=300, bbox_inches='tight')
    print(f"{metric.capitalize()} distribution comparison matrix to", os.path.join('../fig/single_student', teacher_model_name, student_model_name, f'{metric}_distribution_comparison_matrix.png'))
    plt.close()

def plot_performance_matrix(all_output, metric, teacher_model_name, student_model_name):
    matrix_performance = {
        'both_correct': {'positive': 0, 'negative': 0, 'texts': []},
        'teacher_correct_student_wrong': {'positive': 0, 'negative': 0, 'texts': []},
        'teacher_wrong_student_correct': {'positive': 0, 'negative': 0, 'texts': []},
        'both_wrong': {'positive': 0, 'negative': 0, 'texts': []}
    }

    for ex in all_output:
        label = int(ex['label'])
        pred_teacher = int(ex['predicted_label_teacher'][metric])
        pred_student = int(ex['predicted_label_student'][metric])

        if label == 1:
            gt_category = 'positive'
        else:
            gt_category = 'negative'

        # Compare predictions and update the matrix
        if pred_teacher == label and pred_student == label:
            matrix_performance['both_correct'][gt_category] += 1
            matrix_performance['both_correct']['texts'].append(ex['input'])
        elif pred_teacher == label and pred_student != label:
            matrix_performance['teacher_correct_student_wrong'][gt_category] += 1
            matrix_performance['teacher_correct_student_wrong']['texts'].append(ex['input'])
        elif pred_teacher != label and pred_student == label:
            matrix_performance['teacher_wrong_student_correct'][gt_category] += 1
            matrix_performance['teacher_wrong_student_correct']['texts'].append(ex['input'])
        else:
            matrix_performance['both_wrong'][gt_category] += 1
            matrix_performance['both_wrong']['texts'].append(ex['input'])

    total = len(all_output)

    
    proportions = {
        'both_correct': {
            'total': (matrix_performance['both_correct']['positive'] + matrix_performance['both_correct']['negative']) / total,
            'positive': matrix_performance['both_correct']['positive'] / total,
            'negative': matrix_performance['both_correct']['negative'] / total
        },
        'teacher_correct_student_wrong': {
            'total': (matrix_performance['teacher_correct_student_wrong']['positive'] + matrix_performance['teacher_correct_student_wrong']['negative']) / total,
            'positive': matrix_performance['teacher_correct_student_wrong']['positive'] / total,
            'negative': matrix_performance['teacher_correct_student_wrong']['negative'] / total
        },
        'teacher_wrong_student_correct': {
            'total': (matrix_performance['teacher_wrong_student_correct']['positive'] + matrix_performance['teacher_wrong_student_correct']['negative']) / total,
            'positive': matrix_performance['teacher_wrong_student_correct']['positive'] / total,
            'negative': matrix_performance['teacher_wrong_student_correct']['negative'] / total
        },
        'both_wrong': {
            'total': (matrix_performance['both_wrong']['positive'] + matrix_performance['both_wrong']['negative']) / total,
            'positive': matrix_performance['both_wrong']['positive'] / total,
            'negative': matrix_performance['both_wrong']['negative'] / total
        }
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#f7fbff", "#c6dbef", "#6baed6", "#08519c"]
    cmap = LinearSegmentedColormap.from_list("academic_blues", colors)

    matrix = np.array([
        [proportions['both_correct']['total'], proportions['teacher_correct_student_wrong']['total']],
        [proportions['teacher_wrong_student_correct']['total'], proportions['both_wrong']['total']]
    ])

    ax = sns.heatmap(matrix, annot=False, cmap=cmap, linewidths=2, 
                     linecolor='white', cbar=False, square=True)

    ax.set_xticklabels(['Success', 'Fail'], fontsize=14, fontweight='bold')
    ax.set_yticklabels(['Success', 'Fail'], fontsize=14, fontweight='bold')

    ax.set_xlabel('MIA Result on Student Model', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('MIA Result on Teacher Model', fontsize=16, fontweight='bold', labelpad=15)

    cell_texts = [
        [f"{proportions['both_correct']['total']:.3f}\n({proportions['both_correct']['positive']:.3f}, {proportions['both_correct']['negative']:.3f})",
         f"{proportions['teacher_correct_student_wrong']['total']:.3f}\n({proportions['teacher_correct_student_wrong']['positive']:.3f}, {proportions['teacher_correct_student_wrong']['negative']:.3f})"],
        [f"{proportions['teacher_wrong_student_correct']['total']:.3f}\n({proportions['teacher_wrong_student_correct']['positive']:.3f}, {proportions['teacher_wrong_student_correct']['negative']:.3f})",
         f"{proportions['both_wrong']['total']:.3f}\n({proportions['both_wrong']['positive']:.3f}, {proportions['both_wrong']['negative']:.3f})"]
    ]

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, cell_texts[i][j], 
                    ha="center", va="center", fontsize=16,
                    color="black" if matrix[i, j] < 0.25 else "white")

    plt.suptitle(f'Attack Performance Comparison ({metric.capitalize()})', 
                 fontsize=20, fontweight='bold', y=0.975)
    plt.title(f'{teacher_model_name} / {student_model_name}', 
              fontsize=16, pad=15)

    fig.text(0.5, 0.01, 
             "Each cell shows the proportion of training data in each category\n(proportion of member data, proportion of non-member data)",
             ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)

    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)
    plt.savefig(os.path.join('../fig/single_student', teacher_model_name, student_model_name, 
                           f'{metric}_attack_performance_comparison_matrix.png'), 
                dpi=300, bbox_inches='tight')
    print(f"{metric.capitalize()} attack performance comparison matrix to", os.path.join('../fig/single_student', teacher_model_name, student_model_name, f'{metric}_attack_performance_comparison_matrix.png'))
    plt.close()


if __name__ == "__main__":
    args = Options()
    args = args.parser.parse_args()
    
    teacher_model_name = args.teacher_model_name
    teacher_model_testing_results_path = args.teacher_model_testing_results_path
    student_model_names = args.student_model_names
    student_model_testing_results_paths = args.student_model_testing_results_paths
    
    input_path_dict = {
        teacher_model_name: teacher_model_testing_results_path
    }
    for i, model in enumerate(student_model_names):
        input_path_dict[model] = student_model_testing_results_paths[i]

    for student_model_name in student_model_names:
        input_path_student = input_path_dict[student_model_name]
        all_output, optimal_thresholds_teacher, optimal_thresholds_student = read_tables_from_multiple_files(teacher_model_testing_results_path, input_path_student)
        
        teacher_accuracy, student_accuracy, teacher_sd, student_sd, prediction_summary = calculate_metrics(all_output)
        
        for metric in ['recall', 'll', 'zlib', 'mink', 'mink++', 'ref']:
            plot_distribution_matrix(all_output, metric, teacher_model_name, student_model_name)
            plot_performance_matrix(all_output, metric, teacher_model_name, student_model_name)
            