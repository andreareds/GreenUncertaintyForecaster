import argparse
import pandas as pd
import datetime
import numpy as np

# GPU_SPECS = {"P100": (1596, 0.000404575892857, 0.0208),
#              "T4": (994, 0.00056640625, 0.0058),
#              "V100": (1912, 0.000283203125, 0.0233),
#              "MISC": (2240, 0.000361596009975, 0.0208),
#              }
GPU_SPECS = {"P100": (1596, 5821.61047808277, 0.0208),
             "T4": (994, 4158.29319863055, 0.0058),
             "V100": (1912, 8316.5863972611, 0.0233),
             "MISC": (2240, 6513.57645566739, 0.0208),
             }

QOS_LEVELS = ["90", "91", "92", "93", "94", "95", "96", "97", "98", "99"]


def load_results(folder: str) -> dict:
    results = {"MCD": pd.read_csv(f"{folder}/monte_results.csv"),
               "HBNN": pd.read_csv(f"{folder}/hbnn_results.csv"),
               "HBNN++": pd.read_csv(f"{folder}/flbnn_results.csv"),
               "LSTMQ": pd.read_csv(f"{folder}/lstmq_results.csv"),
               "LSTM": pd.read_csv(f"{folder}/lstm_results.csv")
               }
    return results


def allocate_resources_2(w_demand: float,
                         df_allocation: pd.DataFrame,
                         i: int,
                         ) -> float:
    # print(f"{i}: {datetime.datetime.now()}")
    wd_rem = w_demand

    w = np.array(df_allocation.w)
    s = np.array(df_allocation.s)
    e = np.array(df_allocation.e)
    c = np.array(df_allocation.c)

    index = len(s)-1
    cost = 0
    while wd_rem > 0:
        # Otherwise we select the most efficient
        wd_rem -= w[index]
        cost += e[index] + s[index] * c[index]
        index -= 1
    return cost


def run_scenario_2(args, results: dict):
    # build dataframe of GPUs
    w = []  # stores workload that GPUs can provide
    e = []  # stores energy consumption
    c = []  # stores energy cost to turn on GPUs
    s = []  # stores the status of GPUs
    g_name = []  # stores the name of the GPUs
    for gpu_name in GPU_SPECS:
        w_value = GPU_SPECS[gpu_name][1]
        n_g = GPU_SPECS[gpu_name][0]
        e_g = GPU_SPECS[gpu_name][2]
        g_name.extend([gpu_name for i in range(n_g)])
        w.extend([w_value for i in range(n_g)])
        e.extend([e_g for i in range(n_g)])
        c.extend([1 for i in range(n_g)])
        s.extend([0 for i in range(n_g)])
    d = {"gpu_name": g_name, 'w': w, 'e': e, 'c': c, 's': s}
    df_gpus_status = pd.DataFrame(data=d)

    # efficiency of GPU = workload provided / energy required
    df_gpus_status['cost'] = df_gpus_status.e + df_gpus_status.s * df_gpus_status.c
    df_gpus_status['eff'] = df_gpus_status.w / df_gpus_status.cost
    df_gpus_status = df_gpus_status.sort_values(by='eff')
    df_gpus_status.reset_index(inplace=True)

    # load results
    if args.debug_mode:
        # pred_workloads = {"baseline_b": [0.7 for i in range(len(results["HBNN"]))]}
        pred_workloads = {"baseline_a": results["HBNN"]["true_gpu"].values,
                          # "baseline_b": [43916358.2147171-10
                          #                for i in range(len(results["HBNN"]))],  # max possible workload
        #                   "HBNN": results["HBNN"][f"ub_{args.qos_level}"].values[:3],
        #                   "MCD": results["MCD"][f"ub_{args.qos_level}"].values[:3],
        #                   "HBNN++": results["HBNN++"][f"ub_{args.qos_level}"].values[:3],
        #                   "LSTMQ": results["LSTMQ"][f"ub_{args.qos_level}"].values[:3],
        #                   "LSTM": results["LSTM"][f"ub_{args.qos_level}"].values[:3],
                          }
    else:
        pred_workloads = {"baseline_a": results["HBNN"]["true_gpu"].values,
                          "baseline_b": [43916358.2147171 - 10
                                         for i in range(len(results["HBNN"]))],  # max possible workload
                          "HBNN": results["HBNN"][f"ub_{args.qos_level}"].values,
                          "MCD": results["MCD"][f"ub_{args.qos_level}"].values,
                          "HBNN++": results["HBNN++"][f"ub_{args.qos_level}"].values,
                          "LSTMQ": results["LSTMQ"][f"ub_{args.qos_level}"].values,
                          "LSTM": results["LSTM"][f"ub_{args.qos_level}"].values,
                          }

    model_column = []
    energy_costs = []
    qos_column = []
    for qos in QOS_LEVELS:
        print(f"{datetime.datetime.now()} -- SCENARIO {args.id_scenario} for QOS {qos}!")
        scenario_2_costs = {}
        for model in pred_workloads:
            demands = pred_workloads[model]
            tot_costs = [allocate_resources_2(demand, df_gpus_status, i) for i, demand in enumerate(demands)]
            scenario_2_costs[model] = np.sum(tot_costs)
            model_column.append(model)
            energy_costs.append(scenario_2_costs[model])
            qos_column.append(qos)
            print(f"Cost for {model}: {scenario_2_costs[model]}")
            print(f"{datetime.datetime.now()} -- Done with model {model}!")
        print(f"{datetime.datetime.now()} -- END!")

    d = {"qos": qos_column,
         "model": model_column,
         "energy_cost": energy_costs}

    df_results = pd.DataFrame(data=d)
    df_results.to_csv("scenario_2_results.csv", index=False)


def run_scenario_4(results: dict):
    pass


def main(args):

    results = load_results(args.data_folder)

    if args.id_scenario == "2":
        print("RUNNING SCENARIO 2")
        run_scenario_2(args, results)
    if args.id_scenario == "4":
        print("RUNNING SCENARIO 4")
        run_scenario_4(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        default=False,
        type=str,
        required=True,
        help="The path to the dataset"
    )
    parser.add_argument(
        "--id_scenario",
        default=False,
        type=str,
        required=True,
        help="The path to the dataset"
    )
    parser.add_argument(
        "--qos_level",
        default=False,
        type=str,
        required=True,
        help="The qos level"
    )
    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )

    main(parser.parse_args())
