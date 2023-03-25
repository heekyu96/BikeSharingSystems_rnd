# Utils
class _progressbar_printer:
    def __init__(self, runner_name, iterations, fill_length=45, fillout="█", CR=True):
        self.runner_name = runner_name
        self.iters = iterations + 1
        self.fill_len = fill_length
        self.fill = fillout
        self.line_end = {True: "\r", False: "\r\n"}
        self.print_end = self.line_end[CR]
        self.iter_time = 0.0
        self._progressBar(0)
        pass

    def _progressBar(self, cur, iter_t=0.0):
        cur = cur + 1
        time_left = ("{0:.2f}").format(0)
        self.iter_time += iter_t
        time_spend = ("{0:.2f}").format(self.iter_time)
        if not iter_t == 0.0:
            time_left = ("{0:.2f}").format(self.iter_time / cur * self.iters)
            # time_spend = ("{0:.2f}").format(timer()-self.time_stamp)
        percent = ("{0:." + str(1) + "f}").format(100 * (cur / float(self.iters)))
        filledLength = int(self.fill_len * cur // self.iters)
        bar = self.fill * filledLength + "-" * (self.fill_len - filledLength)
        print(
            f"\r{self.runner_name}\t|{bar}| {percent}% {time_spend}/{time_left}s",
            end=self.print_end,
            flush=True,
        )
        # Print New Line on Complete
        if cur >= self.iters:
            print("\nCompleted\n")

# Path config
data_path = "./03.Data/xx.chicago_bike_dataset/"

root_path = "./02.Code/"
code_path = ""
model_weight_path = root_path+"trained_model_weights/"
plot_path = root_path+"event_prediction_results/"