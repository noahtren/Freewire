## Todo
### Performance
* Take care of any low-hanging fruit in bottlenecks
  * the largest bottleneck according to `torch.utils.bottleneck` is `to`, with about 60% of all time.
  * The rest is `IndexBackward`, `IndexPutBackward`, and `_index_put_impl_`
* Investigate why backwards call takes significantly longer than forward
pass. I've heard it should take no longer than 2x.
  * for MNIST, backward pass currently takes more than 3x the length of forward pass

### Experiments
* Experiment with Watts-Strogatz graph generators (small world networks)
* Implement real-time pruning of individual parameters and entire nodes
* Some neuroevolution experiments

### Benchmarks
* Classification on MNIST and CIFAR-10