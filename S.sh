#PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
#q01          up   infinite      4    mix g[01-04]
#q02          up   infinite      6    mix g[19-24]
#q03          up   infinite      1  down* g26
#q03          up   infinite      2    mix g[27,36]
#q03          up   infinite      1   idle g25
#q04          up   infinite      2  down* g37,gg01
#q04          up   infinite      1  drain gg02
#q04          up   infinite      5    mix g[38-42]
#q_ai         up   infinite     21    mix g[05-17,28-35]
#q_ai         up   infinite      1   idle g18




srun -p q04 --nodelist g38 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
srun -p q04 --nodelist g39 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
srun -p q04 --nodelist g40 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
srun -p q04 --nodelist g41 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
srun -p q04 --nodelist g42 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat

#srun -p q04 --nodelist g37 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q04 --nodelist gg01 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q04 --nodelist gg02 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat


#
#srun -p q03 --nodelist g27 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q03 --nodelist g36 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q03 --nodelist g27 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q03 --nodelist g27 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#
#
#
#srun -p q01 --nodelist g01 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q01 --nodelist g02 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q01 --nodelist g03 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q01 --nodelist g04 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#
#srun -p q02 --nodelist g19 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q02 --nodelist g20 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q02 --nodelist g21 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q02 --nodelist g22 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q02 --nodelist g23 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q02 --nodelist g24 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#



#srun -p q04 --nodelist gg01 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
#srun -p q04 --nodelist gg01 --nodes=1 --cpus-per-task=1 --tasks-per-node=1 gpustat
