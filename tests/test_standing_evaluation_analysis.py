import torch
from setup.analyze_standing_evaluation import common_window


def test_common_window_excludes_survivor_extra_time_and_reset_states():
    def frame(active,value):
        return {'active':torch.tensor(active),'tilt_deg':torch.full((4,),float(value)),
                'joint_mse':torch.full((4,),float(value)**2),'body_mse':torch.full((4,),float(value)**2)}
    initial=[frame([True]*4,1),frame([True]*4,2),frame([False,True,True,True],100)]
    final=[frame([True]*4,3),frame([True]*4,4),frame([True]*4,5),frame([True]*4,999)]
    rows=common_window(initial,final)
    assert rows[0]['steps']==2 and rows[0]['tilt_deg_initial_final']==[1.5,3.5]
    assert rows[1]['steps']==3 and rows[1]['tilt_deg_initial_final'][1]==4.
    assert abs(rows[0]['joint_rmse_initial_final'][0]-(2.5**.5))<1e-6
