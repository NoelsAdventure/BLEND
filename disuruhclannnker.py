import json                                                                                                                                                            
with open('trained_models/LoraF_invi_visi_rank_1/test/seperate_mixed_5050_adaptive_gt.json', 'r') as f:                                                                                          
    data = json.load(f)                                                                                                                                                                         
first_step = data['episodes'][0]['steps_data'][0]                                                                                                                                               
print("Robot:", first_step['robot'])                                                                                                                                                       
print("First human:", first_step['humans'][0])  