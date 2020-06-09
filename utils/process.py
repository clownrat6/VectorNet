import os
import numpy as np
import matplotlib.pyplot as plt

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence

root_base_path = 'data'

_sample_path = '{}/{}/argoverse-forecasting/forecasting_sample/data'\
    .format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], root_base_path)


extract_x = lambda list_: [x[0] for x in list_]
extract_y = lambda list_: [x[1] for x in list_]


class argoverse_processor(object):
    def __init__(self, scenario_path = _sample_path):
        self.map_pres = ArgoverseMap()
        self.scenarios = ArgoverseForecastingLoader(scenario_path)
        self.seq_lane_props = self.map_pres.city_lane_centerlines_dict

    def acquire_agent_trajectory(self, one_scenario):
        # There is 5 seconds of one scenario in total. 0.1s interval so that it has 50 items.
        return one_scenario.agent_traj

    def acquire_city(self, one_scenario):
        return one_scenario.seq_df['CITY_NAME'].values[0]

    def acquire_lane_centerlines_and_edgelines(self, one_scenario):
        city_name = self.acquire_city(one_scenario)
        seq_def = one_scenario.seq_df
        seq_lanes = self.seq_lane_props[city_name]
        
        x_min = min(seq_def['X'])
        x_max = max(seq_def['X'])
        y_min = min(seq_def['Y'])
        y_max = max(seq_def['Y'])

        lane_centerlines = []
        lane_ids = []
        
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lanes.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)
                lane_ids.append(lane_id)

        edges1 = []
        edges2 = []

        for lane_id in lane_ids:
            lane_polyline = self.map_pres.get_lane_segment_polygon(lane_id, city_name)
            lane_polyline = lane_polyline[:-1]
            break_point = len(lane_polyline)/2
            break_point = int(break_point)
            edges1.append(lane_polyline[:break_point])
            edges2.append(lane_polyline[break_point:])

        return lane_centerlines, edges1, edges2

    def visualization_lanes(self, one_scenario):
        lane_cls, lane_eds1, lane_eds2 = self.acquire_lane_centerlines_and_edgelines(one_scenario)
        
        for lane_cl in lane_cls:
            a, = plt.plot(lane_cl[:, 0], lane_cl[:, 1], '--', label='lane_centerline', color='#17A9C3', linewidth=2) # color='#C7398D')
        
        for lane_ed in lane_eds1:
            b, = plt.plot(lane_ed[:, 0], lane_ed[:, 1], label='lane_edgeline', color='#C7398D', linewidth=2) # color='#314E87')
        
        for lane_ed in lane_eds2:
            c, = plt.plot(lane_ed[:, 0], lane_ed[:, 1], label='lane_edgeline', color='#C7398D', linewidth=2)
        
        plt.legend([a, b], ['lane_centerline', 'lane_edgeline'], fontsize=12)
        plt.axis('off')
        plt.show()


    def visualization_trajectory(self, one_scenario):
        lane_cls, lane_eds1, lane_eds2 = self.acquire_lane_centerlines_and_edgelines(one_scenario)
        
        for lane_cl in lane_cls:
            plt.plot(lane_cl[:, 0], lane_cl[:, 1], '--', color='#17A9C3', linewidth=2) # color='#C7398D')
        
        for lane_ed in lane_eds1:
            plt.plot(lane_ed[:, 0], lane_ed[:, 1], color='#C7398D', linewidth=2) # color='#314E87')
        
        for lane_ed in lane_eds2:
            plt.plot(lane_ed[:, 0], lane_ed[:, 1], color='#C7398D', linewidth=2)

        traj = self.acquire_agent_trajectory(one_scenario)

        plt.plot(traj[:, 0], traj[:, 1], '-', color='#E0DD00', alpha=1, linewidth=1, zorder=2)
        plt.plot(traj[-1, 0], traj[-1, 1], 'o', color='#E0DD00', alpha=1, markersize=7, zorder=2)
        
        plt.axis('off')
        plt.show()


class scenario_object(object):
    def __init__(self, argoverse_scenario, argoverse_processor):
        self.main = argoverse_scenario
        self.ap = argoverse_processor
        self.lane_cls, self.lane_edges1, self.lane_edges2 = \
            self.ap.acquire_lane_centerlines_and_edgelines(self.main)
        self.lane_edges = []
        for ed1,ed2 in zip(self.lane_edges1, self.lane_edges2):
            ed1 = [(x[0], x[1]) for x in ed1]
            ed2 = [(x[0], x[1]) for x in ed2]
            ed = list(zip(ed1, ed2))
            self.lane_edges.append(np.array(ed))

        self.agent_traj = self.ap.acquire_agent_trajectory(self.main)


def scenario_vectorization(one_scenario):
    agent_traj = one_scenario.agent_traj
    lane_edges = one_scenario.lane_edges

    # (0, 2] used as observation and (2, 5] used as trajectory prediction
    # 0.1 observation interval
    last_observed_coordinate = agent_traj[19]
    lc = last_observed_coordinate

    vector_sets = []

    # Firstly, we use lane edge to vectorize the lane rather than use lane center line.
    for lane_edge in lane_edges:
        # single lane
        one_polyline = []
        # left-hand side lane edge line vectorization
        for i in range(len(lane_edge)-1):
            start = (lane_edge[i][0][0] - lc[0], lane_edge[i][0][1] - lc[1])
            end = (lane_edge[i+1][0][0] - lc[0], lane_edge[i+1][0][1] - lc[1])
            vector = np.array([*start, *end])
            one_polyline.append(vector)
        vector_sets.append(one_polyline)
        # right-hand side lane edge line vectorization
        
        one_polyline = []
        for i in range(len(lane_edge)-1):
            start = (lane_edge[i][1][0] - lc[0], lane_edge[i][1][1] - lc[1])
            end = (lane_edge[i+1][1][0] - lc[0], lane_edge[i+1][1][1] - lc[1])
            vector = np.array([*start, *end])
            one_polyline.append(vector)
        vector_sets.append(one_polyline)
        

    # Then, we vectorize the agent trajectory.
    # (0, 2] used as observation and (2, 5] used as trajectory prediction
    train_trajectory = []
    for i in range(19):
        start = (agent_traj[i][0] - lc[0], agent_traj[i][1] - lc[1])
        end = (agent_traj[i+1][0] - lc[0], agent_traj[i+1][1] - lc[1])
        vector = np.array([*start, *end])
        train_trajectory.append(vector)

    test_trajectory = []
    for i in range(20, 50):
        traj_point = (agent_traj[i][0] - lc[0], agent_traj[i][1] - lc[1])
        test_trajectory.append(np.array(traj_point))
    
    return vector_sets, train_trajectory, test_trajectory


if __name__ == "__main__":
    ap = argoverse_processor()

    scenario = scenario_object(ap.scenarios[0], ap)

    viz_sequence(ap.scenarios[0].seq_df, show=True)

    map_pres, train_trajectory, test_trajectory = scenario_vectorization(scenario)

    print('scenario lane polylines num: {}'.format(len(map_pres))) 
    print('train trajectory vector num: {}\ntest trajectory vector num:{}'.format(len(train_trajectory), len(test_trajectory)))

    print(test_trajectory[0])

    ap.visualization_lanes(ap.scenarios[0])
    ap.visualization_trajectory(ap.scenarios[0])