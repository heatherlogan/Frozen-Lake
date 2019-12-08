import sys
from run_random import *
from run_simple import *
from run_random import *
from run_rl import *


def sort_df(df):

    cols = [x*100 for x in range(1,501)]
    df.columns = cols

    for col in cols:
        df[col] = df[col].apply(lambda x: x.replace("'", '') if x==x else '')
        df[col] = df[col].apply(lambda x: float(x) if x==x else '')

    avgs = df.mean(axis=0)
    df.loc['averages'] = pd.Series(avgs)
    df = df.transpose()
    return df

def get_dataframes(map_name):

    if map_name=='8x8-base':
        random_df = pd.read_csv("output_tables/random_agent_8x8-base.csv").set_index('problem_id')
        simple_df = pd.read_csv("output_tables/simple_agent_8x8-base.csv").set_index('problem_id')
        rl_df = pd.read_csv("output_tables/rl_agent_8x8-base.csv").set_index('problem_id')
    else:
        random_df = pd.read_csv("output_tables/random_agent_4x4-base.csv").set_index('problem_id')
        simple_df = pd.read_csv("output_tables/simple_agent_4x4-base.csv").set_index('problem_id')
        rl_df = pd.read_csv("output_tables/rl_agent_4x4-base.csv").set_index('problem_id')

    return random_df, simple_df, rl_df




if __name__=='__main__':

    results_random = []
    results_simple = []
    results_rl = []

    map_name = sys.argv[1]

    if map_name not in ["8x8-base", "4x4-base"]:
        print('Invalid Map, using 4x4-base map')
        map_name = '4x4-base'

    if map_name=='4x4-base':
        p_id_range = range(0,3)
    else:
        p_id_range = range(0,8)

    # run scripts over every problem id

    heading_str = ("problem_id, total rewards, total failures, avg steps to reward, best-case steps to reward, worst case steps to reward, steps to first goal\n")
    averages_random = np.zeros(6)
    averages_simple = np.zeros(6)
    averages_rl = np.zeros(6)

    random_rewards_per_100 = []
    simple_rewards_per_100 = []
    rl_rewards_per_100 = []

    for i in p_id_range:

        # run random_agent
        print('..Running random agent on problem {}, map {}'.format(i, map_name))
        rewards,failures,avg_steps,best_case,worst_case,first_steps, per_100 = run_senseless_agent(problem_id=i, map=map_name)
        results_random.append([rewards,failures,avg_steps,best_case,worst_case,first_steps])
        averages_random[0] += rewards
        averages_random[1] += failures
        averages_random[2] += avg_steps
        averages_random[3] += best_case
        averages_random[4] += worst_case
        averages_random[5] += first_steps
        random_rewards_per_100.append(per_100)


        # run simple agent
        print('..Running simple agent on problem {}, map {}'.format(i, map_name))
        rewards,failures,avg_steps,best_case,worst_case,first_steps, per_100 = run_simple_agent(problem_id=i, map=map_name)
        results_simple.append([rewards,failures,avg_steps,best_case,worst_case,first_steps])
        averages_simple[0] += rewards
        averages_simple[1] += failures
        averages_simple[2] += avg_steps
        averages_simple[3] += best_case
        averages_simple[4] += worst_case
        averages_simple[5] += first_steps
        simple_rewards_per_100.append(per_100)


        # run reinforcement agent
        print('..Running reinforcement agent on problem {}, map {}'.format(i, map_name))
        rewards,failures,avg_steps,best_case,worst_case,first_steps, per_100 = run_reinforcement_agent(problem_id=i, map=map_name)
        results_rl.append([rewards,failures,avg_steps,best_case,worst_case,first_steps])
        averages_rl[0] += rewards
        averages_rl[1] += failures
        averages_rl[2] += avg_steps
        averages_rl[3] += best_case
        averages_rl[4] += worst_case
        averages_rl[5] += first_steps
        rl_rewards_per_100.append(per_100)
    # save results to text files

    if map_name=="8x8-base":
        averages_random = [num/8 for num in averages_random]
        averages_simple = [num/8 for num in averages_simple]
        averages_rl = [num/8 for num in averages_rl]
    else:
        averages_random = [num/4 for num in averages_random]
        averages_simple = [num/4 for num in averages_simple]
        averages_rl = [num/4 for num in averages_rl]


    file_random = open('output_tables/random_agent_{}.csv'.format(map_name), 'w')
    file_simple = open('output_tables/simple_agent_{}.csv'.format(map_name), 'w')
    file_rl = open('output_tables/rl_agent_{}.csv'.format(map_name), 'w')

    heading_str = ("problem_id, total rewards, total failures, avg steps to reward, best-case steps to reward,"
                   " worst case steps to reward, episodes to first reward\n")

    file_random.write(heading_str)
    file_simple.write(heading_str)
    file_rl.write(heading_str)

    for i, result in enumerate(results_random):
        result_str = str(result).replace('[', '').replace(']', '')
        file_random.write("{}, {}\n".format(i, result_str))
    string_avg_random = ('averages, {}\n'.format(str(averages_random).replace('[', '').replace(']', '')))
    file_random.write(string_avg_random)

    for i, result in enumerate(results_simple):
        result_str = str(result).replace('[', '').replace(']', '')
        file_simple.write("{}, {}\n".format(i, result_str))
    string_avg_simple = ('averages, {}\n'.format(str(averages_simple).replace('[', '').replace(']', '')))
    file_simple.write(string_avg_simple)

    for i, result in enumerate(results_rl):
        result_str = str(result).replace('[', '').replace(']', '')
        file_rl.write("{}, {}\n".format(i, result_str))
    string_avg_rl = ('averages, {}\n'.format(str(averages_rl).replace('[', '').replace(']', '')))
    file_rl.write(string_avg_rl)

    # close files
    file_random.close()
    file_simple.close()
    file_rl.close()

    per_100_file_random = open('output_tables/random_rewards_per_100.csv', 'w')
    per_100_file_simple = open('output_tables/simple_rewards_per_100.csv', 'w')
    per_100_file_rl = open('output_tables/rl_rewards_per_100.csv', 'w')

    counts = []

    for i, rewards in enumerate(random_rewards_per_100):
        per_100_file_random.write("{}\n".format(str(rewards).replace('[', '').replace(']', '')))

    for i, rewards in enumerate(simple_rewards_per_100):
        per_100_file_simple.write("{}\n".format(str(rewards).replace('[', '').replace(']', '')))

    for i, rewards in enumerate(rl_rewards_per_100):
        per_100_file_rl.write("{}\n".format(str(rewards).replace('[', '').replace(']', '')))

    # make figures

    random_df, simple_df, rl_df = get_dataframes(map_name)

    print('\n------------ Random Agent ------------\n', random_df)
    print('\n------------ Simple Agent ------------\n', simple_df)
    print('\n-------------- RL Agent --------------\n', rl_df)




