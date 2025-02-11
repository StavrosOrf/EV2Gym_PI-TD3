import pandas as pd
import numpy as np

from multicopula import EllipticalCopula
import pickle
import time

from matplotlib import pyplot as plt


class TimeSeriesDataAugmentor:
    def __init__(self):
        """
        Initialize the data augmentor with a data manager instance and the selected augmentation model.
        Additional parameters can be set here if required.
        """

        self._load_active_power_data()
        augmented_data = self._sample_data()

        # plot the data
        plt.plot(augmented_data)
        plt.show()

    def _load_active_power_data(self):
        """
        Load active power data from the data manager.
        """
        data_path = "three_month_data.csv"

        df = pd.read_csv(data_path, parse_dates=['date_time'])
        print(f'df initial shape is {df.shape}')

        # Drop the "price" column and any columns related to renewable generation.
        cols_to_drop = [col for col in df.columns
                        if col.startswith('price') or col.startswith('renewable_active_power')]
        df.drop(columns=cols_to_drop, inplace=True)

        df['date_time'] = pd.to_datetime(df['date_time'])

        # Extract the date and time from the date_time column.
        df['day'] = df['date_time'].dt.day_of_week
        df.drop(columns=['date_time'], inplace=True)
        df['timestep'] = df.index % 96

        print(f'df shape is {df.values.shape}')
        df = df[:96]

        new_df = pd.DataFrame()

        for i in range(df.values//96):            
            for row in df.values:
                for i in range(1, 34):
                    day = row[-2]
                    active_power_node_i = row[i]
                    new_df = pd.concat([new_df, pd.DataFrame({'day': [day],f'active_power': [active_power_node_i]})])

        print(f'new_df shape is {new_df.shape}')

        dataset = new_df.values.T
        print(f'dataset shape is {dataset.shape}')

        covariance_ = np.array([[1, -0.6,  0.7],
                                [-0.6,    1, -0.4],
                                [0.7,  -0.4,   1]])
        mean_ = np.array([1, 3, 4])
        data = np.random.multivariate_normal(mean_, covariance_, 5000).T
        print(f'data shape is {data.shape}')

        copula_model = EllipticalCopula(dataset)
        copula_model.fit()
        pickle.dump(copula_model, open('augmentor.pkl', 'wb'))

        # exit()

    def _sample_data(self,
                     n_buses=34,
                     n_steps=96,
                     start_day=0,
                     start_step=0,
                     ):

        augmentor = pickle.load(open('augmentor.pkl', 'rb'))

        timer_start = time.time()

        num_samples = n_buses
        print('The number of samples is', num_samples)

        data = np.zeros((n_steps, n_buses))

        for i in range(n_steps):
            day = (start_day + i // 96) // 7
            timestep = (start_step + i) % 96

            while True:
                augmented_data = augmentor.sample(num_samples,
                                              conditional=True,
                                              variables={'x1': day,
                                                         'x2': timestep},
                                              )
                
                # check for nan
                if not np.isnan(augmented_data).any():
                    data[i] = augmented_data
                    break

        print(f' time to augment data: {time.time() - timer_start}')
        print(f'augmented_data shape is {data.shape}')

        return data

if __name__ == "__main__":

    augmentor = TimeSeriesDataAugmentor()
