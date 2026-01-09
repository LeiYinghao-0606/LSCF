%run main.py --dataset douban-book --encoder MF --train_batch_size 2048 --tau 0.28 --ssl_lambda 1.0 --log rs
%python main.py --dataset douban-book --encoder PPR --train_batch_size 16384 --tau 0.24 --ssl_lambda 1.0 --log rs1

%run main.py --dataset tmall --encoder MF --train_batch_size 4096 --tau 0.24 --ssl_lambda 5.0 --log rs
%python main.py --dataset tmall --encoder MF --train_batch_size 32768 --tau 0.24 --ssl_lambda 5.0 --log test

%run main.py --dataset amazon-book --encoder LightGCN --gcn_layer 3 --train_batch_size 4096 --tau 0.22 --ssl_lambda 5.0 --log rs
%python main.py --dataset amazon-book0 --encoder LightGCN --train_batch_size 32768 --tau 0.2 --ssl_lambda 5.0 --log test --gcn_layer 3
# Example for the new momentum-guided diffusion aggregator
%python main.py --dataset amazon-book --encoder MoDiff --gcn_layer 2 --mom_step 0.7 --mom_beta 0.8 --mom_teleport 0.05 --train_batch_size 4096 --tau 0.22 --ssl_lambda 5.0 --log modiff

# Example for FreqMix-RBC (recommended)
%python main.py --dataset douban-book --encoder FreqMix --gcn_layer 2 --fm_lambda 0.1 --fm_mu 0.8 --fm_step 0.4 --fm_drop_high 0.1 --train_batch_size 4096 --tau 0.22 --ssl_lambda 5.0 --log freqmix

# Example for NFE (Neural Field Evolution)
%python main.py --dataset douban-book --encoder NFE --nfe_steps 3 --nfe_T 1.0 --nfe_psi 0.1 --nfe_energy_lambda 1e-3 --nfe_high_lambda 0.0 --gcn_layer 2 --train_batch_size 4096 --tau 0.22 --ssl_lambda 5.0 --log nfe

# Enable GID-CLR regularization (Graph–ID dual-view contrast)
%python main.py --dataset douban-book --encoder NFE --gid_lambda 0.5 --gid_tau 0.2 --nfe_steps 3 --nfe_T 1.0 --nfe_psi 0.1 --train_batch_size 4096 --tau 0.22 --ssl_lambda 5.0 --log nfe_gid
