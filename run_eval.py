import sys
from run_random import *
from run_simple import *
from run_random import *
from run_rl import *
import matplotlib.pyplot as plt


def sort_df(df):

    cols = [x*100 for x in range(1,101)]
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
        random_df = pd.read_csv("random_agent_8x8-base.csv").set_index('problem_id')
        simple_df = pd.read_csv("simple_agent_8x8-base.csv").set_index('problem_id')
        rl_df = pd.read_csv("rl_agent_8x8-base.csv").set_index('problem_id')
    else:
        random_df = pd.read_csv("random_agent_4x4-base.csv").set_index('problem_id')
        simple_df = pd.read_csv("simple_agent_4x4-base.csv").set_index('problem_id')
        rl_df = pd.read_csv("rl_agent_4x4-base.csv").set_index('problem_id')

    return random_df, simple_df, rl_df


def plot_and_save(df, map_name):

    file_name= 'rl_learning_behaviour_{}.png'.format(map_name)

    ax = df['averages'].plot(label='Averages', figsize=(20,10))
    ax.grid(True)
    plt.xlabel('Number of Episodes', fontsize=12)
    plt.ylabel('Average rewards per 100 Episodes', fontsize=12)
    plt.axvline(x=5000, c='red', label='Training Stops')
    plt.title('Learning Behaviour of Reinformement Agent over 8 Problems\n alpha=0.1 gamma=0.95, reward_hole = -0.5 \n',
              fontsize=17)
    ax.legend(loc=4)
    plt.savefig(file_name)
    print('RL Learning Behaviour plot saved to file: ', file_name)


def bar_charts(random_df, simple_df, rl_df, map_name):

    file_name= 'steps_to_goal_{}.png'.format(map_name)

    avg_steps = pd.DataFrame({'Random': random_df[' avg steps to reward'],
                              'Simple': simple_df[' avg steps to reward'],
                              'RL': rl_df[' avg steps to reward']})

    best_steps = pd.DataFrame({'Random': random_df[' best-case steps to reward'],
                               'Simple': simple_df[' best-case steps to reward'],
                               'RL': rl_df[' best-case steps to reward']})

    worst_steps = pd.DataFrame({'Random': random_df[' worst case steps to reward'],
                                'Simple': simple_df[' worst case steps to reward'],
                                'RL': rl_df[' worst case steps to reward']})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(45, 10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    best_steps.plot(kind='bar', width=0.95, ax=ax1, color=['royalblue', 'mediumorchid', 'mediumturquoise'])

    ax1.set_ylabel('Number of episodes to first goal')
    ax1.set_title('Best-case steps to reward', fontsize=20)
    x_offset = -0.1
    y_offset = 0.2

    for p in ax1.patches:
        b = p.get_bbox()
        val = "{}".format(int(b.y1 + b.y0))
        ax1.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), fontsize='x-large')

    avg_steps.plot(kind='bar', width=0.95, ax=ax2, color=['royalblue', 'mediumorchid', 'mediumturquoise'])

    ax2.set_title('Average steps to reward', fontsize=20)
    x_offset = -0.1
    y_offset = 0.2

    for p in ax2.patches:
        b = p.get_bbox()
        val = "{}".format(int(b.y1 + b.y0))
        ax2.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), fontsize='x-large')

    worst_steps.plot(kind='bar', width=0.95, ax=ax3, color=['royalblue', 'mediumorchid', 'mediumturquoise'])

    ax3.set_title('Worst-case steps to reward', fontsize=20)
    x_offset = -0.1
    y_offset = 0.2

    for p in ax3.patches:
        b = p.get_bbox()
        val = "{}".format(int(b.y1 + b.y0))
        ax3.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), fontsize='x-large')

    fig.savefig(file_name)

    print('Number of steps to reward plot saved to file: ', file_name)

    #   LINE GRAPH OF AWARDS ACHIEVED

    all_rewards = pd.DataFrame({'Random': random_df[' total rewards'],
                                'Simple': simple_df[' total rewards'],
                                'RL': rl_df[' total rewards']})

    file_name_2 = 'awards_achieved_{}.png'.format(map_name)
    all_rewards_percent = (all_rewards / 10000) * 100
    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    all_rewards_percent.plot(kind='line', color=['royalblue', 'mediumorchid', 'mediumturquoise'], ax=axes2)
    axes2.set_ylabel('Episodes Rewards were Achieved')
    axes2.set_title('Percentage of Rewards Achieved in 10000 episodes')
    axes2.legend(loc=1)
    file_name_2 = 'awards_achieved_{}.png'.format(map_name)
    fig2.savefig(file_name_2)
    print('Percentage of Rewards Achieved in 10000 episodes saved to file, ', file_name_2)


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
        print(averages_random)
        averages_random = [num/8 for num in averages_random]
        averages_simple = [num/8 for num in averages_simple]
        averages_rl = [num/8 for num in averages_rl]
    else:
        averages_random = [num/3 for num in averages_random]
        averages_simple = [num/3 for num in averages_simple]
        averages_rl = [num/3 for num in averages_rl]


    file_random = open('random_agent_{}.csv'.format(map_name), 'w')
    file_simple = open('simple_agent_{}.csv'.format(map_name), 'w')
    file_rl = open('rl_agent_{}.csv'.format(map_name), 'w')

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

    per_100_file_random = open('random_rewards_per_100_{}.csv'.format(map_name), 'w')
    per_100_file_simple = open('simple_rewards_per_100_{}.csv'.format(map_name), 'w')
    per_100_file_rl = open('rl_rewards_per_100_{}.csv'.format(map_name), 'w')

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

    print('Agent results saved to csv files')
    per_100_df = sort_df(pd.DataFrame(rl_rewards_per_100))
    plot_and_save(per_100_df, map_name)
    bar_charts(random_df, simple_df, rl_df, map_name)


