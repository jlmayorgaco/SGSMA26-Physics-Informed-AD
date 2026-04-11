

# Simulation S0: Faults Scenari
faults = [
    fault( node = 12, type = 0, t0=0, tf = 1000s, params = {....}),
    fault( node = 8, type = 2, t0=11, tf = 1000s, params = {....}),
    fault( node = 8, type = 2, t0=11, tf = 1000s, params = {....}),
    fault( node = 8, type = 2, t0=11, tf = 1000s, params = {....}),
    fault( node = 8, type = 2, t0=11, tf = 1000s, params = {....}),
    fault( node = 8, type = 2, t0=11, tf = 1000s, params = {....}),
    fault( node = 8, type = 2, t0=11, tf = 1000s, params = {....}),
]

simulation = Simulation_IEEE_39_BUS_Andes()
simulation.set_faults(faults)
simulation.run()

nodes_pmu = simulation.get_pmu_nodes()
nodes_non_pmu = simulation.get_non_pmu_nodes()

len(nodes_pmu) + len(nodes_non_pmu) == 39

# Nodes With PMU
node_8 = simulation.get_by_node_id(8)
node_8 = simulation.get_by_node_id(8)
node_8 = simulation.get_by_node_id(8)
node_8 = simulation.get_by_node_id(8)
node_8 = simulation.get_by_node_id(8)
node_8 = simulation.get_by_node_id(8)

# Nodes Withou PMU
node_5 = simulation.get_by_node_id(5)
...
...

# Function that has swing equation for nodes, power flow, kirfooc, Ybus, conection bettwen nods (Topologies),etc
# This model can infere the state of a node using data from other models
model = IEEE39_Model()
estimator = IEEE39_Estimator() # Basic Estimator
#estimator = IEEE39_Estimator_Kalman() #  Estimator usng the model to run a EKF o UKF to handle stocahsi noise 

# set pmus nodes with the "real sensed data"
estimator.setNode1(node_1)
estimator.setNode2(node_2)
estimator.setNode3(node_3)
estimator.setNode4(node_4)
estimator.setNode8(node_8)

estimator.estimate()
estimated_node_12 = estimator.get_by_node_id(12)

# Phase 1.5 Trainnign
Detector
Classificator 
trianig con set de datos sintenticos generarados por estimator. para un set de casos, genrar muchisimos casos para modeo bien entrando. 



#PHASE 2 Detection
detector= Detector() #ML model training to get a time series of a window N, and detct is data is normal o faul
[faults_detections_1] = detector.detect(estimated_node_1)
[faults_detections_2] = detector.detect(estimated_node_2)...
# run detections for all nodes
# faults_detections_1 has a array of when detected a fault, and timerange of that fuakl, a node timesires can have multiples faults or non

#Phase 3 Classsifciacion
classificator= Classificator() #ML model training to get a time series of a window N, and detct is data is normal o faul
[faults_classifcations1] = classificator.detect(estimated_node_1, faults_detections_1)
[faults_classifcations2] = classificator.detect(estimated_node_2, faults_detections_2)
[faults_classifcations3] = classificator.detect(estimated_node_3, faults_detections_3)
[faults_classifcations4] = classificator.detect(estimated_node_4, faults_detections_4)
# Here using the time ranges and the timesitne classifcation detect which fault is it using a ML model 
...
....

metrics = Metrics()
m_rmse_node_12 = metrics.get_rmse(estimated = estimated_node_12, real = simulation.get_by_node_id(12))
m_rmse_node_12 = metrics.get_rmse(estimated = estimated_node_12, real = simulation.get_by_node_id(12))
.....

m_accuray
..
..
..
othjers metris rletad with ML , false positves, traninig accuracty, etc etc. model whieh

mtric dedicate de la guideline que nos rankea. 

plotter = Plotter()
plotter.plot_diagram()
plotter.plot_voltage_a_by_node_id(1)
plotter.plot_voltage_3f_by_node_id(1)

plotter....
otros plots valiosos para el programar....



cuando se corra este archivo, dos param traning op run , training do training o retriangi, run only get the existing models.