import aifeynman

pathdir = "/home/mk422/Documents/DAC_SR/SR_algorithms/Ai_feynman/"
filename = "GTLeadingOnes.txt"
BF_try_time = 60
BF_ops_file_type = "14ops.txt"
polyfit_deg = 3
NN_epochs = 500
vars_name = ['x1', 'x2', 'pi']
test_percentage = 20

aifeynman.run_aifeynman(pathdir, filename, BF_try_time= BF_try_time, BF_ops_file_type=BF_ops_file_type, polyfit_deg=polyfit_deg, NN_epochs=NN_epochs, vars_name=vars_name, test_percentage=test_percentage)