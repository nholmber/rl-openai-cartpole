import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

def main():
	# Data obtained with python cartpole_policy_gradient.py --fixed-seed

	# default reward_method
	plain = np.loadtxt('plain.txt')

	# reward_method='to-go'
	togo = np.loadtxt('reward-to-go.txt')

	# state-value function used as baseline + reward_method='to-go'
	value = np.loadtxt('value.txt')

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	plt.plot(plain, label='Plain')
	plt.plot(togo, label='Reward-to-go')
	plt.plot(value, label='Value')
	ax.set_xlim([0,30])
	ax.set_ylim([0,200])
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Average reward")
	plt.legend()
	plt.savefig('policy.png', bbox_inches='tight', format='png', dpi=300)
	#plt.show()

if __name__ == '__main__':
	main()