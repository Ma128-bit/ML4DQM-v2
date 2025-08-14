import os, json, subprocess, argparse, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utilities.database as db
from sklearn.metrics import confusion_matrix

class MEPerformance:
    def __init__(self, job_id, ring, anomalysample):
        self.ring = ring
        self.station = ""
        self.good_data_path = glob.glob(db.get_job_info(job_id)["S2"])
        if anomalysample==None:
            self.anomaly_data_path = db.get_job_info(job_id)["S3"]
        else:
            self.anomaly_data_path =  anomalysample
        self.me_name = db.get_job_info(job_id)["MEname"]
        self.metrics = ["max_loss", "min_loss"]
        self.direction = ["up", "down"]
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id) + os.sep + self.me_name+'_'+ring+'_step4'
        self.good_data = None
        self.anomaly_data = None
    
    def update_jobid(self, job_id):
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id) + os.sep + self.me_name+'_'+ring+'_step4'

    def load_data(self):
        self.good_data = pd.concat( [pd.read_parquet(f, engine="pyarrow", columns=self.metrics) for f in self.good_data_path], ignore_index=True )
        self.anomaly_data = pd.read_parquet(self.anomaly_data_path, engine='pyarrow', columns=self.metrics)

    def compute_performance(self, metic, direction):
        good_data_percentile_5 = np.percentile(self.good_data[metic], 0)
        good_data_percentile_95 = np.percentile(self.good_data[metic], 99)

        anomaly_data_percentile_5 = np.percentile(self.anomaly_data[metic], 0)
        anomaly_data_percentile_95 = np.percentile(self.anomaly_data[metic], 99)

        ran = (min(good_data_percentile_5, anomaly_data_percentile_5), max(good_data_percentile_95, anomaly_data_percentile_95))
        #ran = (min(np.min(self.good_data[metic]), np.min(self.anomaly_data[metic])), max(np.max(self.good_data[metic]), np.max(self.anomaly_data[metic])))
        bin_edges_good = np.linspace(ran[0], ran[1], 81)
        counts_good, _ = np.histogram(self.good_data[metic], bins=bin_edges_good, density=True)
        counts_bad, _ = np.histogram(self.anomaly_data[metic], bins=bin_edges_good, density=True)

        if direction == "up":
            n_good_selected = [
                len(self.good_data[self.good_data[metic] > threshold]) for threshold in bin_edges_good
            ]
            n_bad_selected = [
                len(self.anomaly_data[self.anomaly_data[metic] > threshold]) for threshold in bin_edges_good
            ]
        else:  # 'lower'
            n_good_selected = [
                len(self.good_data[self.good_data[metic] < threshold]) for threshold in bin_edges_good
            ]
            n_bad_selected = [
                len(self.anomaly_data[self.anomaly_data[metic] < threshold]) for threshold in bin_edges_good
            ]

        total_bad = self.anomaly_data.shape[0]
        total_good = self.good_data.shape[0]
        tp = np.array(n_bad_selected)
        fp = np.array(n_good_selected)
        fn = np.array([total_bad - n_bad_selected[i] for i in range(len(n_bad_selected))])
        tn = np.array([total_good - n_good_selected[i] for i in range(len(n_good_selected))])

        f1 = 2 * tp / (2 * tp + fp + fn)

        threshold = bin_edges_good[np.argmax(f1)]

        fig, ax = plt.subplots()
        ax.bar(bin_edges_good[:-1], counts_good, width=np.diff(bin_edges_good), align='edge', edgecolor='blue', label=f"{self.station} Data (GOOD)")
        ax.bar(bin_edges_good[:-1], counts_bad, width=np.diff(bin_edges_good), align='edge', edgecolor='red', label=f"{self.station} Simulations (BAD)")
        ax.axvline(threshold, color="black", linestyle="--", label="Best Threshold")
        ax.set_xlabel(metic)
        ax.set_ylabel('a.u.')
        ax.legend()
        fig.savefig(f'{self.output_path}_'+metic+'_compariosn_.png')

        # Compute metrics:
        f1_score = f1[np.argmax(f1)]
        tp_at_threshold = tp[np.argmax(f1)]
        fp_at_threshold = fp[np.argmax(f1)]
        fn_at_threshold = fn[np.argmax(f1)]
        tn_at_threshold = tn[np.argmax(f1)]
        

        precision = tp_at_threshold / (tp_at_threshold + fp_at_threshold)
        accuracy = (tp_at_threshold + tn_at_threshold) / (
            tp_at_threshold + tn_at_threshold + fp_at_threshold + fn_at_threshold
        )
        tpr = tp_at_threshold / (tp_at_threshold + fn_at_threshold)
        fpr = fp_at_threshold / (fp_at_threshold + tn_at_threshold)

        columns = ['metric','threshold', 'f1_score', 'precision', 'accuracy', 'tpr', 'fpr', 
           'tp_at_threshold', 'fp_at_threshold', 'fn_at_threshold', 'tn_at_threshold']
        df_metrics = pd.DataFrame(columns=columns)

        df_metrics.loc[0] = [metic, threshold, f1_score, precision, accuracy, tpr, fpr, tp_at_threshold, fp_at_threshold, fn_at_threshold, tn_at_threshold]

        return df_metrics, threshold

    def performance_matrix(self, thresholds):
        df_good = self.good_data[self.metrics].copy()
        df_bad = self.anomaly_data[self.metrics].copy()
        df_good["gen_label"] = 0
        df_bad["gen_label"] = 1
        df_good["pred_label"] = 0
        df_bad["pred_label"] = 0
        for i, metic in enumerate(self.metrics):
            if self.direction[i] == "up":
                df_good['pred_label'] = df_good[metic].apply(lambda x: 1 if x > thresholds[i] else 0)
                df_bad['pred_label'] = df_bad[metic].apply(lambda x: 1 if x > thresholds[i] else 0)
            elif self.direction[i] == "down":
                df_good['pred_label'] = df_good[metic].apply(lambda x: 1 if x < thresholds[i] else 0)
                df_bad['pred_label'] = df_bad[metic].apply(lambda x: 1 if x < thresholds[i] else 0)
        df = pd.concat([df_good, df_bad])
        cm = confusion_matrix(df['gen_label'], df['pred_label'])
        tn, fp, fn, tp = cm.ravel()
        return f"fp: {fp} fn:{fn} tn:{tn} tp:{tp}"

    def run(self):
        self.load_data()
        df_temp = []
        thresholds = []
        for i, metic in enumerate(self.metrics):
            df, threshold = self.compute_performance(metic, self.direction[i])
            df_temp.append(df)
            thresholds.append(threshold)
        df_metrics = pd.concat(df_temp)
        df_metrics = df_metrics.reset_index(drop=True)
        columns = ['f1_score', 'precision', 'accuracy', 'tpr', 'fpr']
        out_string = ""
        for index, row in df_metrics.iterrows():
            print(index)
            out_string += f"{self.metrics[index]} - "
            for c in columns:
                out_string += c+f": {row[c]} "
            out_string += " - "
        out_string += self.performance_matrix(thresholds)
        with open(f'{self.output_path}_performance.txt', "w") as file:
            file.write(out_string)
        df_metrics.to_csv(f'{self.output_path}_metrics.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True, help="job_id")
    parser.add_argument("--ring", type=str, required=True, help="in or out")
    parser.add_argument("--anomalysample", type=str, required=False, help="OPTIONAL: path to an external anomaly sample")
    
    args = parser.parse_args()
    job_id = args.job_id
    ring = args.ring

    analyzer = MEPerformance(job_id, ring, args.anomalysample)

    change_jobid = False
    job_id_new, _ = db.update_step(job_id, "S4", analyzer.output_path)
    if job_id_new != job_id:
        analyzer.update_jobid(job_id_new)
        db.update_step(job_id_new, "S4", analyzer.output_path)
        change_jobid = True
    
    analyzer.run()
    
    if change_jobid:
        print("!!! Your new job id is: ", job_id_new)
