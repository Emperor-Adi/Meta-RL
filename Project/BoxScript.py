import numpy as np

Boxes = {
	"CartPole-v1": 
	{
		"Expected Reward": 475,

		"Deterministic":
		[
			10.0, 
			0.5,
			0.1
		],

		"Random":
		[
			np.random.uniform(5.0, 15.0),
			np.random.uniform(0.25, 0.75),
			np.random.uniform(0.05, 0.5)
		],

		"Extreme":
		[
			np.random.choice(
			[
				np.random.uniform(1.0,5.0),
				np.random.uniform(15.0,20.0)
			]),
			np.random.choice(
			[
				np.random.uniform(0.05,0.25),
				np.random.uniform(0.75,1.0)
			]),
			np.random.choice(
			[
				np.random.uniform(0.01,0.05),
				np.random.uniform(0.5,1.0)
			])
		]
	},

	"MountainCar-v0":
	{
		"Expected Reward": -145,

		"Deterministic":
		[
			0.001,
			0.0025
		],
		"Random":
		[
			np.random.uniform(0.0005,0.005),
			np.random.uniform(0.001,0.005)
		],
		"Extreme":
		[
			np.random.choice(
			[
				np.random.uniform(0.0001,0.0005),
				np.random.uniform(0.005,0.01)
			]),
			np.random.choice(
			[
				np.random.uniform(0.0005,0.001),
				np.random.uniform(0.005,0.01)
			])
		]
	},

	"Acrobot-v1":
	{
		"Expected Reward": -100,

		"Deterministic":
		[
			1,
			1,
			1
		],
		"Random":
		[
			np.random.uniform(0.75,1.25),
			np.random.uniform(0.75,1.25),
			np.random.uniform(0.75,1.25)
		],
		"Extreme":
		[
			np.random.choice(
			[
				np.random.uniform(0.5,0.75),
				np.random.uniform(1.25,1.5)
			]),
			np.random.choice(
			[
				np.random.uniform(0.5,0.75),
				np.random.uniform(1.25,1.5)
			]),
			np.random.choice(
			[
				np.random.uniform(0.5,0.75),
				np.random.uniform(1.25,1.5)
			])
		]
	},

	"Pendulum-v1":
	{
		"Expected Reward": -100,

		"Deterministic":
		[
			1,
			1
		],
		"Random":
		[
			np.random.uniform(0.75,1.25),
			np.random.uniform(0.75,1.25)
		],
		"Extreme":
		[
			np.random.choice(
			[
				np.random.uniform(0.5,0.75),
				np.random.uniform(1.25,1.5)
			]),
			np.random.choice(
			[
				np.random.uniform(0.5,0.75),
				np.random.uniform(1.25,1.5)
			])
		]
	},

	"HalfCheetah-v3":
	{
		"Expected Reward": 4800,

		"Deterministic":
		[
			0.90,
			1000,
			0.8
		],
		"Random":
		[
			np.random.uniform(0.70,1.10),
			np.random.uniform(750,1250),
			np.random.uniform(0.5,1.1)
		],
		"Extreme":
		[
			np.random.choice(
			[
				np.random.uniform(0.50,0.70),
				np.random.uniform(1.10,1.30)
			]),
			np.random.choice(
			[
				np.random.uniform(500,750),
				np.random.uniform(1250,1500)
			]),
			np.random.choice(
			[
				np.random.uniform(0.2,0.5),
				np.random.uniform(1.1,1.4)
			])
		]
	},

	"Hopper-v3":
	{
		"Expected Reward": 3800,

		"Deterministic":
		[
			0.75,
			1000,
			0.8
		],
		"Random":
		[
			np.random.uniform(0.60,0.90),
			np.random.uniform(750,1250),
			np.random.uniform(0.5,1.1)
		],
		"Extreme":
		[
			np.random.choice(
			[
				np.random.uniform(0.40,0.60),
				np.random.uniform(0.90,1.10)
			]),
			np.random.choice(
			[
				np.random.uniform(500,750),
				np.random.uniform(1250,1500)
			]),
			np.random.choice(
			[
				np.random.uniform(0.2,0.5),
				np.random.uniform(1.1,1.4)
			])
		]
	}
}