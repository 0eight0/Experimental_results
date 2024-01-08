import collections
import os

baseline_logs = ['log_ecommerce_baseline','log_ontonotes_baseline','log_msra_baseline','log_resume_baseline']
ecommerce_logs = ['log_ecommerce_attention','log_ecommerce_attention_no_contrastive','log_ecommerce_gate_newself','log_ecommerce_gate_oldself','log_ecommerce_gate_oldself_spanlen','log_ecommerce_gate_newself_no_contrastive','log_ecommerce_gate_oldself_no_contrastive']
all_logs = [item for item in os.listdir('logs/log_nolite_real') if item.startswith('log')]
all_logs.sort()
'''acc: 0.5789 - recall: 0.0070 - f1: 0.0139 - loss: 24.1353 '''
output = collections.defaultdict(list)
output_acc = collections.defaultdict(list)
output_recall = collections.defaultdict(list)
for log in all_logs:
    # if 'baseline' not in log and '0107' not in log:
    #     continue
# for log in baseline_logs:
    with open(os.path.join('logs/log_nolite_real',log),'r') as f:
        for line in f:
            if ' - loss:' in line:
                acc = float(line.split("acc:")[1][1:7])
                recall = float(line.split("recall:")[1][1:7])
                F1 = float(line.split("f1:")[1][1:7])
                output[log].append(F1)
                output_acc[log].append(acc)
                output_recall[log].append(recall)
# print(output)
for log,F1_list in output.items():
    acc_list = output_acc[log]
    recall_list = output_recall[log]
    print(len(F1_list),'log:{:<50}\tavg-acc:{:.5f}\tavg-recall:{:.5f}\tavg-F1:{:.5f}\tmax-F1:{}'.format(log,(sum(acc_list[-10:])/len(acc_list[-10:])),(sum(recall_list[-10:])/len(recall_list[-10:])),(sum(F1_list[-10:])/len(F1_list[-10:])),max(F1_list)))
