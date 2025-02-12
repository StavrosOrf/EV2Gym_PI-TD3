import pandas as pd
import numpy as np

from multicopula import EllipticalCopula
import pickle
import time

from matplotlib import pyplot as plt


class TimeSeriesDataAugmentor:
    def __init__(self):

        self._load_active_power_data()

    def _load_active_power_data(self):
        """
        Load active power data from the data manager.
        """
        data_path = "three_month_data.csv"

        df = pd.read_csv(data_path, parse_dates=['date_time'])
        print(f'df initial shape is {df.shape}')

        # Drop the "price" column and any columns related to renewable generation.
        # cols_to_drop = [col for col in df.columns
        #                 if col.startswith('price') or col.startswith('renewable_active_power')]
        cols_to_drop = [col for col in df.columns
                        if col.startswith('price') or col.startswith('active_power')]
        df.drop(columns=cols_to_drop, inplace=True)

        df['date_time'] = pd.to_datetime(df['date_time'])

        # Extract the date and time from the date_time column.
        df['day'] = df['date_time'].dt.day_of_week
        df.drop(columns=['date_time'], inplace=True)
        df['timestep'] = df.index % 96

        print(f'df shape is {df.values.shape}')
        # df = df[:96*7]
        new_df = pd.DataFrame()
        print(f'number of days is {int(len(df.values)//96)}')

        for j in range(int(len(df.values)//96)):
            # for row in df.values[j*96:(j+1)*96]:
            for i in range(1, 34):
                day = df.values[j*96, -2].astype(int)
                active_power_96_node_i = df.values[j*96:(j+1)*96, i]
                entry = {'day': [day], }

                for k in range(96):
                    entry[f'active_power_{k}'] = active_power_96_node_i[k]

                new_df = pd.concat(
                    [new_df, pd.DataFrame(entry)], ignore_index=True)

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
                     n_buses: int,
                     n_steps: int,
                     start_day: int,
                     start_step: int = 0,
                     ):

        augmentor = pickle.load(open('augmentor.pkl', 'rb'))

        timer_start = time.time()

        data = np.zeros((int(np.ceil((start_step + n_steps)/96))*96, n_buses))

        for j in range(int(np.ceil((start_step + n_steps)/96))):
            day = (start_day + j) % 7
            while True:
                augmented_data = augmentor.sample(n_buses,
                                                  conditional=True,
                                                  variables={'x1': day},
                                                  )

                if not np.isnan(augmented_data).any():
                    data[j*96:(j+1)*96, :] = augmented_data
                    break

        print(f'time to augment data: {time.time() - timer_start}')
        print(f'augmented_data shape is {data.shape}')

        return data[start_step:start_step+n_steps, :]


if __name__ == "__main__":

    augmentor = TimeSeriesDataAugmentor()

    augmented_data = augmentor._sample_data(n_buses=34,
                                            n_steps=420,
                                            start_day=5,
                                            start_step=50,
                                            )

    # plot the data
    plt.plot(augmented_data[:,:4])
    plt.show()
