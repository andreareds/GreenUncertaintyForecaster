import argparse
import pandas as pd
import datetime
import numpy as np


# model: (# GPUs, comp capacity, max consumption, # GPUs per machine, total # machines)
GPU_SPECS = {"P100": (1596, 5821.61047808277, 0.0208, 2, 798),
             "T4": (994, 4158.29319863055, 0.0058, 2, 497),
             "V100": (1912, 8316.5863972611, 0.0233, 8, 239),
             "MISC": (2240, 6513.57645566739, 0.0208, 8, 280),
             }

AVG_GPU_SPECS = (6742, 6513.57645566739, 0.0208)

QOS_LEVELS = ["90", "91", "92", "93", "94", "95", "96", "97", "98", "99"]
# QOS_LEVELS = ["99"]


def load_results(folder: str) -> dict:
    results = {"MCD": pd.read_csv(f"{folder}/monte_results.csv"),
               "HBNN": pd.read_csv(f"{folder}/hbnn_results.csv"),
               "HBNN++": pd.read_csv(f"{folder}/flbnn_results.csv"),
               "LSTMQ": pd.read_csv(f"{folder}/lstmq_results.csv"),
               "LSTM": pd.read_csv(f"{folder}/lstm_results.csv"),
               "PROPHET": pd.read_csv(f"{folder}/prophet_results.csv")
               }
    return results


def allocate_resources_22(w_demand: float,
                          real_demand: float,
                          df_allocation: pd.DataFrame,
                          i: int,
                          predictor_name: str,
                          ) -> (float, pd.DataFrame):
    # print(f"{i}: {datetime.datetime.now()}")

    names = df_allocation.gpu_name
    w = np.array(df_allocation.w)
    s = np.array(df_allocation.s)
    e = np.array(df_allocation.e)
    c = np.array(df_allocation.c)
    w_u = np.array(df_allocation.w_u)
    e_u = np.array(df_allocation.e_u)

    cost = 0
    index = len(w)-1

    # if overpredicting
    if w_demand > real_demand:
        w_over = w_demand - real_demand
        wd_rem = real_demand

        # accomodate real demand first
        while wd_rem > 0:

            if wd_rem - w[index] < 0:  # I can't fill the entire server
                cost += c[index] + c[index] * 0.2 * s[index]  # I pay to turn on the server
                cost += int(wd_rem/w_u[index]) * e_u[index]  # I pay to turn on some GPUs
            else:
                cost += e[index] + c[index] + c[index] * 0.2 * s[index]  # pay for the real demand

            wd_rem -= w[index]
            s[index] = 0
            index -= 1

        # then pay for turning on more servers with the remaining demand, but GPUs stay in sleep mode
        # wd_rem += w_over
        while w_over > 0 and index >= 0:
            w_over -= w[index]

            # you don't pay for computations
            cost += c[index] + c[index] * 0.2 * s[index]  # pay for turning ON the server (s[index] should be 1)
            s[index] = 0  # server goes ON
            index -= 1

    else:  # underprediction, oracle, naive
        wd_rem = w_demand
        # accomodate the predicted demand, and some real demand will be unmet
        while wd_rem > 0:
            wd_rem -= w[index]
            cost += e[index] + c[index] + c[index] * 0.2 * s[index]  # pay for the real demand
            s[index] = 0
            index -= 1

        if predictor_name == "naive":  # turn ON all servers
            while index >= 0:
                cost += c[index] * 0.2 * s[index]  # cost to turn on the server if off
                cost += c[index]  # cost to keep the server ON with no computation
                s[index] = 0  # you should turn on the server if this is OFF
                index -= 1

    if predictor_name != "naive":
        # all the first servers (unutilised) need to be switched OFF
        off = np.array(list(map(lambda x: 1, s[:index+1])))
        s = np.append(off, s[index+1:])

    d = {"gpu_name": list(names), 'w': list(w), 'e': list(e), 's': list(s), 'c': list(c), 'w_u': w_u, 'e_u': e_u}
    df_gpus_status = pd.DataFrame(data=d)

    # sort by efficiency
    df_gpus_status['eff'] = df_gpus_status.w / (df_gpus_status.e + df_gpus_status.c +
                                                df_gpus_status.c * 0.2 * df_gpus_status.s)

    df_gpus_status = df_gpus_status.sort_values(by='eff')
    df_gpus_status.reset_index(inplace=True)

    return cost, df_gpus_status


def allocate_resources_33(w_demand: float,
                          real_demand: float,
                          df_allocation: pd.DataFrame,
                          i: int,
                          predictor_name: str,
                          ) -> (float, pd.DataFrame):
    # print(f"{i}: {datetime.datetime.now()}")

    names = df_allocation.gpu_name
    w = np.array(df_allocation.w)
    s = np.array(df_allocation.s)
    e = np.array(df_allocation.e)
    c = np.array(df_allocation.c)
    w_u = np.array(df_allocation.w_u)
    e_u = np.array(df_allocation.e_u)

    cost = 0
    index = len(w)-1

    # if overpredicting
    if w_demand > real_demand:
        w_over = w_demand - real_demand
        wd_rem = real_demand

        # accomodate real demand first
        while wd_rem > 0:

            if wd_rem - w[index] < 0:  # I can't fill the entire server
                cost += c[index] + c[index] * 0.2 * s[index]  # I pay to turn on the server
                cost += int(wd_rem/w_u[index]) * e_u[index]  # I pay to turn on some GPUs
            else:
                cost += e[index] + c[index] + c[index] * 0.2 * s[index]  # pay for the real demand

            wd_rem -= w[index]
            s[index] = 0
            index -= 1

        # then pay for turning on more servers with the remaining demand, and GPUs go idling
        # wd_rem += w_over
        while w_over > 0 and index >= 0:
            w_over -= w[index]

            # you don't pay for computations
            cost += c[index] + c[index] * 0.2 * s[index] + e[index]*0.2  # pay for turning ON the server (s[index] should be 1)
            s[index] = 0  # server goes ON
            index -= 1

    else:  # underprediction, oracle, naive
        wd_rem = w_demand
        # accomodate the predicted demand, and some real demand will be unmet
        while wd_rem > 0:
            wd_rem -= w[index]
            cost += e[index] + c[index] + c[index] * 0.2 * s[index]  # pay for the real demand
            s[index] = 0
            index -= 1

        if predictor_name == "naive":  # turn ON all servers
            while index >= 0:
                cost += c[index] * 0.2 * s[index]  # cost to turn on the server if off
                cost += c[index] + e[index] * 0.2  # cost to keep the server ON with no computation
                s[index] = 0  # you should turn on the server if this is OFF
                index -= 1

    if predictor_name != "naive":
        # all the first servers (unutilised) need to be switched OFF
        off = np.array(list(map(lambda x: 1, s[:index+1])))
        s = np.append(off, s[index+1:])

    d = {"gpu_name": list(names), 'w': list(w), 'e': list(e), 's': list(s), 'c': list(c), 'w_u': w_u, 'e_u': e_u}
    df_gpus_status = pd.DataFrame(data=d)

    # sort by efficiency
    df_gpus_status['eff'] = df_gpus_status.w / (df_gpus_status.e + df_gpus_status.c +
                                                df_gpus_status.c * 0.2 * df_gpus_status.s)

    df_gpus_status = df_gpus_status.sort_values(by='eff')
    df_gpus_status.reset_index(inplace=True)

    return cost, df_gpus_status


def run_scenario_22(args, results: dict):
    # build dataframe of GPUs
    w = []  # stores the workload that servers can provide if ON at the beginning of the time window
    w_u = []  # the unitary workload of a GPU
    e = []  # stores energy consumption of GPUS on the server when ON
    e_u = []  # the unitary energy sunsumption of a GPU
    c = []  # stores the cost to keep ON the server
    s = []  # stores the status of GPUs (0=ON, 1=OFF)
    g_name = []  # stores the name of the GPUs
    for gpu_name in GPU_SPECS:
        w_value = GPU_SPECS[gpu_name][1]
        n_g = GPU_SPECS[gpu_name][4]
        m_g = GPU_SPECS[gpu_name][3]
        e_g = GPU_SPECS[gpu_name][2]*0.7  # the max consumption is reduced in average
        c_g = 0.0017
        g_name.extend([gpu_name for i in range(n_g)])
        w.extend([w_value*m_g for i in range(n_g)])
        w_u.extend(([w_value for i in range(n_g)]))
        e.extend([e_g*m_g for i in range(n_g)])
        e_u.extend([e_g for i in range(n_g)])
        s.extend([1 for i in range(n_g)])  # (0=ON, 1=OFF)
        c.extend([c_g for i in range(n_g)])
    d = {"gpu_name": g_name, 'w': w, 'e': e, 's': s, 'c': c, 'w_u': w_u, 'e_u': e_u}
    df_gpus_status = pd.DataFrame(data=d)

    # efficiency of the server = workload provided / energy required to turn them ON and accomodate demand
    df_gpus_status['eff'] = df_gpus_status.w / (df_gpus_status.e + df_gpus_status.c * 1.2 * df_gpus_status.s)
    df_gpus_status = df_gpus_status.sort_values(by='eff')
    df_gpus_status.reset_index(inplace=True)

    model_column = []
    energy_costs = []
    qos_column = []
    for qos in QOS_LEVELS:
        # load results
        if args.debug_mode:
            pred_workloads = {"naive": results["HBNN"]["true_gpu"].values,
                              "jack": [43916358.2147171 - 10 for i in range(len(results["HBNN"]))],
                              }
        else:
            pred_workloads = {"oracle": results["HBNN"]["true_gpu"].values,
                              "naive": results["HBNN"]["true_gpu"].values,  # but you treat this differently
                              "HBNN": results["HBNN"][f"ub_{qos}"].values,
                              "MCD": results["MCD"][f"ub_{qos}"].values,
                              "HBNN++": results["HBNN++"][f"ub_{qos}"].values,
                              "LSTMQ": results["LSTMQ"][f"ub_{qos}"].values,
                              "LSTM": results["LSTM"][f"ub_{qos}"].values,
                              "PROPHET": results["PROPHET"][f"ub_{qos}"].values,
                              }

        print(f"{datetime.datetime.now()} -- SCENARIO {args.id_scenario} for QOS {qos}!")
        scenario_22_costs = {}
        for model in pred_workloads:
            demands = pred_workloads[model]
            tot_costs = 0
            new_gpus_status = df_gpus_status.copy()
            array_cost = []
            real_demand = results["HBNN"]["true_gpu"].values
            for i, demand in enumerate(demands):
                cost_i, new_gpus_status = allocate_resources_22(demand, real_demand[i], new_gpus_status, i, model)
                tot_costs += cost_i
                array_cost.append(cost_i)
            scenario_22_costs[model] = np.sum(tot_costs)
            model_column.append(model)
            energy_costs.append(scenario_22_costs[model])
            qos_column.append(qos)
            print(f"Cost for {model}: {scenario_22_costs[model]}")
            print(f"{datetime.datetime.now()} -- Done with model {model}!")
        print(f"{datetime.datetime.now()} -- END!")

    d = {"qos": qos_column,
         "model": model_column,
         "energy_cost": energy_costs}

    df_results = pd.DataFrame(data=d)
    df_results.to_csv("scenario_22_results.csv", index=False)


def run_scenario_33(args, results: dict):
    # build dataframe of GPUs
    w = []  # stores the workload that servers can provide if ON at the beginning of the time window
    w_u = []  # the unitary workload of a GPU
    e = []  # stores energy consumption of GPUS on the server when ON
    e_u = []  # the unitary energy sunsumption of a GPU
    c = []  # stores the cost to keep ON the server
    s = []  # stores the status of GPUs (0=ON, 1=OFF)
    g_name = []  # stores the name of the GPUs
    for gpu_name in GPU_SPECS:
        w_value = GPU_SPECS[gpu_name][1]
        n_g = GPU_SPECS[gpu_name][4]
        m_g = GPU_SPECS[gpu_name][3]
        e_g = GPU_SPECS[gpu_name][2]*0.7  # the max consumption is reduced in average
        c_g = 0.0017
        g_name.extend([gpu_name for i in range(n_g)])
        w.extend([w_value*m_g for i in range(n_g)])
        w_u.extend(([w_value for i in range(n_g)]))
        e.extend([e_g*m_g for i in range(n_g)])
        e_u.extend([e_g for i in range(n_g)])
        s.extend([1 for i in range(n_g)])  # (0=ON, 1=OFF)
        c.extend([c_g for i in range(n_g)])
    d = {"gpu_name": g_name, 'w': w, 'e': e, 's': s, 'c': c, 'w_u': w_u, 'e_u': e_u}
    df_gpus_status = pd.DataFrame(data=d)

    # efficiency of the server = workload provided / energy required to turn them ON and accomodate demand
    df_gpus_status['eff'] = df_gpus_status.w / (df_gpus_status.e + df_gpus_status.c * 1.2 * df_gpus_status.s)
    df_gpus_status = df_gpus_status.sort_values(by='eff')
    df_gpus_status.reset_index(inplace=True)

    model_column = []
    energy_costs = []
    qos_column = []
    for qos in QOS_LEVELS:
        # load results
        if args.debug_mode:
            pred_workloads = {"naive": results["HBNN"]["true_gpu"].values,
                              "jack": [43916358.2147171 - 10 for i in range(len(results["HBNN"]))],
                              }
        else:
            pred_workloads = {"oracle": results["HBNN"]["true_gpu"].values,
                              "naive": results["HBNN"]["true_gpu"].values,  # but you treat this differently
                              "HBNN": results["HBNN"][f"ub_{qos}"].values,
                              "MCD": results["MCD"][f"ub_{qos}"].values,
                              "HBNN++": results["HBNN++"][f"ub_{qos}"].values,
                              "LSTMQ": results["LSTMQ"][f"ub_{qos}"].values,
                              "LSTM": results["LSTM"][f"ub_{qos}"].values,
                              "PROPHET": results["PROPHET"][f"ub_{qos}"].values,
                              }

        print(f"{datetime.datetime.now()} -- SCENARIO {args.id_scenario} for QOS {qos}!")
        scenario_33_costs = {}
        for model in pred_workloads:
            demands = pred_workloads[model]
            tot_costs = 0
            new_gpus_status = df_gpus_status.copy()
            array_cost = []
            real_demand = results["HBNN"]["true_gpu"].values
            for i, demand in enumerate(demands):
                cost_i, new_gpus_status = allocate_resources_33(demand, real_demand[i], new_gpus_status, i, model)
                tot_costs += cost_i
                array_cost.append(cost_i)
            scenario_33_costs[model] = np.sum(tot_costs)
            model_column.append(model)
            energy_costs.append(scenario_33_costs[model])
            qos_column.append(qos)
            print(f"Cost for {model}: {scenario_33_costs[model]}")
            print(f"{datetime.datetime.now()} -- Done with model {model}!")
        print(f"{datetime.datetime.now()} -- END!")

    d = {"qos": qos_column,
         "model": model_column,
         "energy_cost": energy_costs}

    df_results = pd.DataFrame(data=d)
    df_results.to_csv("scenario_33_results.csv", index=False)


def main(args):

    results = load_results(args.data_folder)

    if args.id_scenario == "22":
        print("RUNNING SCENARIO PERSISTENCE MODE=OFF")
        run_scenario_22(args, results)
    if args.id_scenario == "33":
        print("RUNNING SCENARIO PERSISTENCE MODE=ON")
        run_scenario_33(args, results)


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
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )

    main(parser.parse_args())
