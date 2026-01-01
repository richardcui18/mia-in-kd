import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import json
from analysis_options import Options
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def read_tables_from_file(input_path):
    """
    Read results tables from CSV file and reconstruct the original data structure
    """
    csv.field_size_limit(sys.maxsize)
    all_output = {}
    current_metric = None

    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        
        for row in reader:
            if not row:
                continue
            
            if row[0].startswith("Table for Metric:"):
                # Extract metric and thresholds
                metric = row[0].split("Metric: ")[1]
                current_metric = metric
                
            elif row[0] == "Data Point":
                continue
                
            else:
                # Extract data point information
                input_key = row[0]
                label = int(row[1])
                pred_teacher_value = float(row[2])
                pred_student_value = float(row[3])
                
                # Check if this data point already exists in all_output
                if input_key in all_output:
                    existing_data_point = all_output[input_key]
                    existing_data_point["pred_teacher"][current_metric] = pred_teacher_value
                    existing_data_point["pred_student"][current_metric] = pred_student_value
                else:
                    # Add new data point to all_output
                    if 'bert' in input_path.lower():
                        all_output[input_key] = {
                        "input": input_key,
                        "label": label,
                        "pred_teacher": {current_metric: pred_teacher_value, 'recall': 0},
                        "pred_student": {current_metric: pred_student_value, 'recall': 0}
                        }
                    else:
                        all_output[input_key] = {
                            "input": input_key,
                            "label": label,
                            "pred_teacher": {current_metric: pred_teacher_value},
                            "pred_student": {current_metric: pred_student_value}
                        }
    
    all_output_list = list(all_output.values())

    bert_in_model = 'bert' in input_path.lower()
    
    return classify_data_points(all_output_list, bert_in_model)

def read_tables_from_multiple_files(input_path_teacher, input_path_student):
    all_output_teacher, optimal_thresholds_teacher = analysis_ensemble.read_tables_from_file_one_model(input_path_teacher)
    all_output_student, optimal_thresholds_student = analysis_ensemble.read_tables_from_file_one_model(input_path_student)

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

def classify_data_points(all_output, bert_in_model):
    metrics = all_output[0]["pred_teacher"].keys()

    # calculate the optimal threshold for each metric
    optimal_thresholds_teacher = {}
    optimal_thresholds_student = {}
    for metric in metrics:
        if bert_in_model and metric == 'recall':
            optimal_thresholds_teacher[metric] = 0
            optimal_thresholds_student[metric] = 0
        else:
            for ex in all_output:
                if np.isnan(ex["pred_teacher"][metric]).all():
                    ex["pred_teacher"][metric] = 0
            for ex in all_output:
                if np.isnan(ex["pred_student"][metric]).all():
                    ex["pred_student"][metric] = 0
            
            scores_teacher = [ex["pred_teacher"][metric] for ex in all_output]
            scores_student = [ex["pred_student"][metric] for ex in all_output]
            labels = [ex["label"] for ex in all_output]
            optimal_thresholds_teacher[metric] = analysis_ensemble.calculate_optimal_threshold(labels, scores_teacher)
            optimal_thresholds_student[metric] = analysis_ensemble.calculate_optimal_threshold(labels, scores_student)

    # calculate predicted labels to each example based on all metrics
    for ex in all_output:
        ex["predicted_label_teacher"] = {}
        for metric in metrics:
            if ex["pred_teacher"][metric] >= optimal_thresholds_teacher[metric]:
                ex["predicted_label_teacher"][metric] = 1
            else:
                ex["predicted_label_teacher"][metric] = 0
        
        ex["predicted_label_student"] = {}
        for metric in metrics:
            if ex["pred_student"][metric] >= optimal_thresholds_student[metric]:
                ex["predicted_label_student"][metric] = 1
            else:
                ex["predicted_label_student"][metric] = 0

    return all_output, optimal_thresholds_teacher, optimal_thresholds_student

def calculate_metrics(all_output, teacher_model_name, student_model_name):
    true_labels = []
    teacher_recall_preds = []
    teacher_ll_preds = []
    teacher_zlib_preds = []
    student_recall_preds = []
    student_ll_preds = []
    student_zlib_preds = []

    for ex in all_output:
        true_labels.append(int(ex['label']))
        teacher_recall_preds.append(int(ex['predicted_label_teacher']['recall']))
        teacher_ll_preds.append(int(ex['predicted_label_teacher']['loss']))
        teacher_zlib_preds.append(int(ex['predicted_label_teacher']['zlib']))
        student_recall_preds.append(int(ex['predicted_label_student']['recall']))
        student_ll_preds.append(int(ex['predicted_label_student']['loss']))
        student_zlib_preds.append(int(ex['predicted_label_student']['zlib']))
    
    def get_accuracy_and_sd(true, pred):
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tpr + tnr) / 2
        sd_tpr = np.sqrt(tpr * (1 - tpr) / (tp + fn)) if (tp + fn) > 0 else 0
        sd_tnr = np.sqrt(tnr * (1 - tnr) / (tn + fp)) if (tn + fp) > 0 else 0
        sd = 0.5 * np.sqrt(sd_tpr**2 + sd_tnr**2)
        return acc, sd

    # Calculate accuracy and SD
    teacher_recall_acc, teacher_recall_sd = get_accuracy_and_sd(true_labels, teacher_recall_preds)
    teacher_ll_acc, teacher_ll_sd = get_accuracy_and_sd(true_labels, teacher_ll_preds)
    teacher_zlib_acc, teacher_zlib_sd = get_accuracy_and_sd(true_labels, teacher_zlib_preds)
    student_recall_acc, student_recall_sd = get_accuracy_and_sd(true_labels, student_recall_preds)
    student_ll_acc, student_ll_sd = get_accuracy_and_sd(true_labels, student_ll_preds)
    student_zlib_acc, student_zlib_sd = get_accuracy_and_sd(true_labels, student_zlib_preds)

    teacher_accuracy = [teacher_recall_acc, teacher_ll_acc, teacher_zlib_acc]
    student_accuracy = [student_recall_acc, student_ll_acc, student_zlib_acc]
    teacher_sd = [teacher_recall_sd, teacher_ll_sd, teacher_zlib_sd]
    student_sd = [student_recall_sd, student_ll_sd, student_zlib_sd]
    
    prediction_summary = [true_labels, teacher_recall_preds, teacher_ll_preds, teacher_zlib_preds, student_recall_preds, student_ll_preds, student_zlib_preds]

    # Export accuracy and SD
    if teacher_model_name == 'BERT' or teacher_model_name == 'BERT_not_vulnerable':
        labels = ['Loss', 'Zlib']
        teacher_accuracy = teacher_accuracy[1:]
        student_accuracy = student_accuracy[1:]
        teacher_sd = teacher_sd[1:]
        student_sd = student_sd[1:]
        x = range(len(labels))
    else:
        labels = ['ReCall', 'Loss', 'Zlib']
        x = range(len(labels))
    bar_width = 0.35

    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)

    # Create accuracy output text file
    accuracy_output_path = os.path.join('../fig/single_student', teacher_model_name, student_model_name)
    
    accuracy_data = {
        "teacher_model": teacher_model_name,
        "student_model": student_model_name,
        "metrics": []
    }
    
    for i, label in enumerate(labels):
        accuracy_data["metrics"].append({
            "MIA Method": label,
            "teacher_accuracy": {
                "mean": round(float(teacher_accuracy[i]),3),
                "sd": round(float(teacher_sd[i]),3),
                "ci_95": f"({teacher_accuracy[i]-1.96*teacher_sd[i]:.3f}, {teacher_accuracy[i]+1.96*teacher_sd[i]:.3f})"
            },
            "student_accuracy": {
                "mean": round(float(student_accuracy[i]),3),
                "sd": round(float(student_sd[i]),3),
                "ci_95": f"({student_accuracy[i]-1.96*student_sd[i]:.3f}, {student_accuracy[i]+1.96*student_sd[i]:.3f})"
            }
        })
    
    # Write to JSON file
    json_output_path = os.path.join(accuracy_output_path, f'accuracy_values.json')
    print("Accuracy values json file to", json_output_path)
    with open(json_output_path, 'w') as f:
        json.dump(accuracy_data, f, indent=4)

    return teacher_accuracy, student_accuracy, teacher_sd, student_sd, prediction_summary

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
        
        teacher_accuracy, student_accuracy, teacher_sd, student_sd, prediction_summary = calculate_metrics(all_output, teacher_model_name, student_model_name)
        
        for metric in ['recall', 'loss', 'zlib']:
            if metric == 'recall' and teacher_model_name == "BERT":
                continue
            plot_distribution_matrix(all_output, metric, teacher_model_name, student_model_name)
            plot_performance_matrix(all_output, metric, teacher_model_name, student_model_name)
            