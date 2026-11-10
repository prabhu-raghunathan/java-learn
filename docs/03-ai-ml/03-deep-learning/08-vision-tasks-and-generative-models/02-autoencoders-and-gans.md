---
title: "Autoencoders and GANs: Learning to Rebuild and Learning to Fake"
sidebar_position: 2
---

import {AnomalyThresholdLab} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
An **autoencoder** is a network trained to **reconstruct its own input**: an **encoder** squeezes the input into a small **latent code** and a **decoder** rebuilds it, and the **reconstruction loss** (usually MSE) is the only training signal, so no labels are needed. The **bottleneck** forces it to keep only the essentials: on 8 × 8 handwritten digits, an 8-number code rebuilds test images with MSE 0.0145 (PCA with 8 components: 0.0246; the mean image: 0.0740). Uses include compression, **denoising** and **anomaly detection** (high reconstruction error = unusual; AUC 0.88 for spotting an unseen digit). A **GAN** trains two networks against each other: a **generator** turns random noise into fake samples, a **discriminator** tries to tell real from fake, and the generator improves by fooling it. Training is delicate: with balanced learning rates a GAN learned all 8 modes of a toy dataset, but with the discriminator learning 100× faster the generator **mode-collapsed** onto 2 of 8, producing good-looking samples (93% on a real mode) with almost no variety.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Object Detection and Segmentation](./01-object-detection-and-segmentation.md), the other half of the same revision round.
**This page:** the generative-model part of the **Module III revision question bank** (questions 44–47, awareness level in the course), expanded into full notes and verified with small PyTorch models: autoencoders trained on scikit-learn's 8 × 8 digits and GANs trained on a 2-D ring of 8 Gaussian blobs.
**Next:** the [Deep Learning Interview Question Bank](../09-interview-question-bank.md), all 51 revision questions with model answers.
:::

## 🧭 Start Here

Almost every model in this course so far is **supervised**: it learns from pairs of an input and a correct answer (an image and its label, a review and its sentiment). Labels are expensive, and much of the world's data has none. This page is about two families of models that learn from the data itself. An **autoencoder** learns a compact description of the data by practising to rebuild it. A **GAN** (generative adversarial network) learns to *produce* new data that looks like the real thing. Both are examples of **unsupervised** (or self-supervised) and **generative** learning.

The page follows the order you would teach it. It starts with what an autoencoder is (Chapter 1), how it is trained with a **reconstruction loss** (Chapter 2), why the **bottleneck** is the whole point and what the learned **latent space** looks like (Chapter 3), and what autoencoders are used for: denoising and anomaly detection (Chapter 4). A short chapter on the **variational autoencoder** (Chapter 5) bridges to generation. Then it covers GANs: the two-player idea (Chapter 6), how training works and what "balance" looks like in numbers (Chapter 7), and the classic failure, **mode collapse** (Chapter 8). It ends by comparing autoencoders, VAEs, GANs and diffusion models (Chapter 9).

Everything is verified by small real runs on a CPU (PyTorch 2.15, scikit-learn 1.9.1). The autoencoders train on scikit-learn's digits dataset: 1,797 grey-scale images of 8 × 8 pixels, split into 1,347 training and 450 test images. The GANs train on a 2-D toy dataset, 8 Gaussian blobs on a circle, small enough that you can see on a plot exactly which "modes" the generator has learned.

| New concept | One-line meaning |
|---|---|
| **Unsupervised learning** | Learning structure from data without labels |
| **Autoencoder** | Network trained to output its own input through a narrow middle |
| **Encoder / decoder** | Compress the input to a code / rebuild the input from the code |
| **Latent code (z)** | The small vector in the middle: the learned summary of the input |
| **Bottleneck** | The narrow latent layer that stops the network from simply copying |
| **Reconstruction loss** | How different the rebuilt output is from the input (MSE) |
| **Denoising autoencoder** | Fed a noisy input, trained to output the clean one |
| **Anomaly detection** | Flag inputs the autoencoder rebuilds badly |
| **VAE** | Autoencoder whose latent space is a smooth distribution you can sample |
| **GAN** | A generator and a discriminator trained in competition |
| **Generator / discriminator** | Turns noise into fakes / scores real vs fake |
| **Mode collapse** | The generator covers only a few kinds of output |

:::analogy[A sketch artist and a forger]
An autoencoder is a sketch artist who glances at a face, writes down a few notes ("long nose, round glasses"), then redraws the face from the notes alone; the fewer notes allowed, the more they must learn what matters. A GAN is an art forger and an inspector: the forger keeps painting fakes, the inspector keeps spotting them, and both get better until the fakes pass.
:::

## 1. What an Autoencoder Is

An **autoencoder** is a neural network whose target output is its own input. It has two parts. The **encoder** maps the input $\data{x}$ (here 64 pixel values) to a much smaller vector $\pred{z}$, the **latent code** or latent representation (here 2, 8 or 16 numbers). The **decoder** maps $\pred{z}$ back to a full-size output $\pred{\hat x}$ of 64 pixels. Training adjusts both parts so that $\pred{\hat x}$ is as close to $\data{x}$ as possible. Because the "label" is the input itself, no human labelling is needed; this is sometimes called **self-supervised** learning.

The interesting part is not the output, which is just an approximate copy of the input, but the code in the middle. To rebuild 64 pixels from only 8 numbers, the network must discover what those 64 pixels have in common across the dataset: stroke shapes, slants, loops. The encoder becomes a learned **compressor** and feature extractor, and the decoder a learned **decompressor**. The model on this page is small: encoder $64 \to 32 \to k$ with a ReLU in between, decoder $k \to 32 \to 64$ with a **sigmoid** at the end so outputs stay in the 0–1 pixel range; with $k = 8$ it has 4,744 parameters.

![Autoencoder: a 64-pixel input goes through an encoder (64 to 32) to a latent code of k numbers, then a decoder (k to 32) rebuilds 64 pixels; the reconstruction loss compares output with input; test MSE per pixel 0.0358 for k = 2, 0.0145 for k = 8, 0.0076 for k = 16, 0.0740 for the mean image](/img/deep-learning/diagrams/autoencoder.svg)

*How to read it:* data flows left to right and the boxes' heights show the sizes. The dashed red line is the training signal: it compares the output with the input, with no labels anywhere.

<Formula name="Autoencoder" tag="Encode, then decode" words="The encoder maps the input to a small code, the decoder maps the code back to an output of the input's size; both are learned together." remember="Squeeze, then rebuild.">

$$
\begin{aligned}
\pred{z} &= f_{\param{\theta}}(\data{x}) \in \mathbb{R}^{\hyper{k}} \\
\pred{\hat x} &= g_{\param{\phi}}(\pred{z}) \in \mathbb{R}^{d}
\end{aligned}
$$

- $\data{x}$: the input ($d = 64$ pixels); $\pred{\hat x}$: the reconstruction
- $f_{\param{\theta}}$: encoder with weights $\param{\theta}$; $g_{\param{\phi}}$: decoder with weights $\param{\phi}$
- $\hyper{k}$: the latent size you choose (2, 8, 16 here), much smaller than $d$

**Example:** with $\hyper{k} = 8$, each 64-pixel digit becomes 8 numbers; a batch of 450 test digits gives a latent tensor of shape (450, 8).

</Formula>

:::tip[🎯 Interview angle]
**"What is an autoencoder, and what are its two main parts?"** A network trained to reconstruct its input. The encoder compresses the input into a lower-dimensional latent representation; the decoder rebuilds the input from that representation. It learns without labels, and the useful product is the compact latent code, which captures the data's main structure.
:::

## 2. Reconstruction Loss: The Only Training Signal

The autoencoder is trained by minimising a **reconstruction loss**: a number that is small when $\pred{\hat x}$ looks like $\data{x}$ and large when it doesn't. For real-valued inputs such as pixel intensities the standard choice is the **mean squared error (MSE)**: square the difference at every pixel and average. For inputs in 0–1 that behave like probabilities (black-and-white images), binary cross-entropy per pixel is also common. Training is the ordinary PyTorch loop: forward pass, loss, `backward()`, optimizer step; the only unusual line is that the target is the input.

On the digits, the 8-number autoencoder's training MSE falls from 0.0858 after the first epoch to 0.0269 after 10, 0.0151 after 50 and 0.0116 after 300 epochs. On the 450 **test** digits, which it never saw, its MSE is **0.0145**. A useful sanity check is a trivial baseline: always outputting the average training digit gives a test MSE of 0.0740, so the autoencoder removes about 80% of that error. As the course notes say, a low reconstruction loss means the input was reconstructed well, which is what makes the error usable as a signal later (Chapter 4).

<Formula name="Reconstruction loss (MSE)" tag="Autoencoder training" words="Average over all pixels of the squared difference between the input and its reconstruction. Zero means a perfect copy." remember="Rebuilt minus original, squared, averaged.">

$$
\loss{L}(\data{x}, \pred{\hat x}) = \frac{1}{d} \sum_{i=1}^{d} (\data{x_i} - \pred{\hat x_i})^2 = \frac{1}{d} \lVert \data{x} - \pred{\hat x} \rVert^2
$$

- $d$: number of input values (64 pixels); the loss is averaged over a batch as well
- the course writes it as $\lVert \data{x} - \pred{\hat x} \rVert^2$, the same quantity without the $1/d$ averaging

**Example:** a pixel that should be 0.8 and is rebuilt as 0.6 contributes $(0.2)^2 = 0.04$. A test MSE of 0.0145 corresponds to a typical pixel error of about $\sqrt{0.0145} \approx 0.12$ on the 0–1 scale.

</Formula>

![Ten test digits and their reconstructions with 2, 8 and 16 latent numbers; MSE 0.03578, 0.01447 and 0.00757](/img/deep-learning/vision-generative/ae-reconstructions.png)

*How to read it:* the top row is the input; each row below is the same digits rebuilt through a smaller or larger bottleneck. With 2 numbers the 2 in the first column comes back looking like a 3 and the 4s lose their shape; with 16 they are close to the originals.

<Animation src="generative/autoencoder-bottleneck" caption="The same test digit squeezed through 16, then 8, then 2 latent numbers. The rebuilt image loses detail as the bottleneck narrows, and the test error rises from 0.0076 to 0.0358." />

:::tip[🎯 Interview angle]
**"What is reconstruction loss and how is it used?"** It measures how different the reconstruction $\pred{\hat x}$ is from the input $\data{x}$, typically MSE (or binary cross-entropy for 0–1 data). The autoencoder is trained to minimise it, and it's the only training signal, so no labels are needed. After training, a high reconstruction error on a new input signals that the input is unlike the training data, which is the basis of anomaly detection.
:::

## 3. The Bottleneck and the Latent Space

Why make the middle layer small? If the latent code were as large as the input, the easiest solution would be to **copy**: the encoder passes the pixels through unchanged and the decoder passes them back, giving near-zero loss and learning nothing about digits. Making the code narrower than the input (an **undercomplete** autoencoder) forbids copying, so the network has to find a compact description that works for all training images. The size of the bottleneck is a trade-off: the smaller it is, the more general the features but the blurrier the reconstructions.

The measurements show the trade-off directly. With 2, 8 and 16 latent numbers the test MSE is 0.0358, 0.0145 and 0.0076. It is also instructive to compare with **PCA**, the classic linear method that keeps the $k$ directions of greatest variance. An autoencoder with only linear layers and MSE learns the same subspace as PCA; the nonlinear (ReLU) autoencoder does better at every size: 0.0358 vs 0.0530 for $k = 2$, 0.0145 vs 0.0246 for $k = 8$, 0.0076 vs 0.0115 for $k = 16$. With a 64-number "bottleneck" (no compression at all) the test MSE drops further to 0.0057, which looks better but means the model is no longer forced to summarise.

| Latent size $\hyper{k}$ | Autoencoder test MSE | PCA test MSE | Compression |
|---|---|---|---|
| 2 | 0.0358 | 0.0530 | 32 × |
| 8 | 0.0145 | 0.0246 | 8 × |
| 16 | 0.0076 | 0.0115 | 4 × |
| 64 (no bottleneck) | 0.0057 | n/a | none |
| always the mean image | 0.0740 | | |

The **latent space** is where the useful structure lives. Training the 2-number autoencoder and plotting each test digit's code as a point shows that digits of the same class land near each other, even though the model never saw a single label: 2s and 7s spread into regions of their own and 4s form an arc along the top, while 0s, 5s, 6s and 8s crowd together near the centre and 3s overlap with 9s. Two numbers aren't enough to separate all ten digits, which matches the higher reconstruction error at $k = 2$. That is why autoencoders are used for **dimensionality reduction**, visualisation and as pretrained feature extractors.

![2-number latent codes of 450 test digits, coloured by their true digit (unused in training): same digits cluster together](/img/deep-learning/vision-generative/ae-latent-2d.png)

*How to read it:* each dot is one test image placed by its two latent numbers; colours and the digit labels at each cluster's median were added afterwards for reading only.

:::danger[Interview Trap: "lower reconstruction error is always better"]
A bigger bottleneck lowers the error (0.0057 with no compression) but stops the network from learning a summary, and an autoencoder that reconstructs *everything* well is useless for anomaly detection. The goal is a useful representation, not a perfect copy.
:::

:::analogy[Packing for a trip with a small bag]
With an unlimited suitcase you'd throw everything in and learn nothing about what you need. A small backpack forces you to choose the essentials, and that choice is exactly the knowledge the autoencoder is after. Too small, and you arrive without things you needed (blurry reconstructions).
:::

:::tip[🎯 Interview angle]
**"Why does an autoencoder need a bottleneck?"** Without it, the network can learn the identity function and copy the input, learning nothing. A latent size smaller than the input forces it to find the main factors of variation. On 8 × 8 digits, 8 latent numbers rebuild test digits with MSE 0.0145, beating 8-component PCA (0.0246) because the layers are nonlinear; a linear autoencoder would match PCA.
:::

## 4. What Autoencoders Are Used For

**Denoising.** A **denoising autoencoder** is fed a corrupted input and trained to output the *clean* original. It can't succeed by copying, because the noise isn't in the target, so it must learn what real digits look like and project the noisy input back onto them. With Gaussian noise of standard deviation 0.3 added to every pixel, the noisy test digits have MSE 0.0524 against the clean ones; after the 16-number denoising autoencoder, the MSE is 0.0237, less than half.

![Denoising autoencoder: noisy inputs, denoised outputs and clean targets for six test digits; MSE 0.05237 to 0.02374](/img/deep-learning/vision-generative/ae-denoising.png)

**Anomaly detection.** An autoencoder trained only on **normal** data learns to rebuild normal data well. Something unusual, which it never learned to describe, tends to come back badly, so the **reconstruction error itself becomes an anomaly score**. To test this, an 8-number autoencoder was trained on digits 0–8 only, and digit 9 played the anomaly. On the test set, the average error was 0.0143 for normal digits and 0.0309 for 9s, more than twice as high. Ranked by error, a random 9 scores higher than a random normal digit 88.4% of the time (ROC AUC 0.8836). To turn the score into an alarm you pick a threshold: at the 95th percentile of normal errors (0.03098), it catches 17 of the 45 nines while raising false alarms on 21 of 405 normal digits. The distributions overlap, so the threshold is a trade-off that depends on what a missed anomaly costs.

<Formula name="Reconstruction-error anomaly score" tag="Anomaly detection" words="Score each new input by how badly the autoencoder trained on normal data rebuilds it; flag it when the score passes a threshold chosen on normal validation data." remember="Rebuilt badly means unusual.">

$$
\begin{aligned}
s(\data{x}) &= \lVert \data{x} - g_{\param{\phi}}(f_{\param{\theta}}(\data{x})) \rVert^2 / d \\
\text{anomaly} &\iff s(\data{x}) > \hyper{\tau}
\end{aligned}
$$

- $s(\data{x})$: the reconstruction error of one input; $\hyper{\tau}$: the alarm threshold
- a common choice of $\hyper{\tau}$ is a high percentile of the scores on normal validation data

**Example:** with $\hyper{\tau} = 0.03098$ (95th percentile of normal errors), 17 of 45 nines are flagged (recall 37.8%) and 21 of 405 normal digits (false-alarm rate 5.2%).

</Formula>

<AnomalyThresholdLab />

![Histogram of reconstruction errors: digits 0 to 8 cluster at low error; digit 9 spreads to higher error; threshold at 0.03098; AUC 0.8836](/img/deep-learning/vision-generative/ae-anomaly.png)

| Use | How the autoencoder is used | Number on this page |
|---|---|---|
| Compression / dimensionality reduction | Keep the encoder's latent code | 64 → 8 numbers, MSE 0.0145 |
| Visualisation | 2-number code as a scatter plot | digits cluster without labels |
| Denoising | Train noisy input → clean target | MSE 0.0524 → 0.0237 |
| Anomaly detection | Reconstruction error as a score | AUC 0.8836 for unseen digit 9 |
| Pretraining / features | Reuse the encoder in a supervised model | (not run here) |

:::tip[🎯 Interview angle]
**"How can an autoencoder detect anomalies?"** Train it only on normal data, then use the reconstruction error as an anomaly score: inputs unlike the training data are rebuilt badly. Choose the threshold on normal validation data. Trained on digits 0–8, the error for unseen 9s was about twice as high on average (0.0309 vs 0.0143), with AUC 0.88; overlap means a threshold always trades missed anomalies against false alarms.
:::

## 5. From Rebuilding to Generating: the Variational Autoencoder

A plain autoencoder can't really **generate** new data. Its decoder only knows how to decode the codes the encoder actually produced, and those codes sit in scattered clusters with gaps between them. Pick a random point in the latent space and decode it, and you usually get noise, because nothing taught the decoder what to do there.

A **variational autoencoder (VAE)** fixes that by making the latent space a smooth probability distribution. The encoder outputs, for each input, a mean $\pred{\mu}$ and a standard deviation $\pred{\sigma}$ for every latent dimension instead of a single point, and the code is **sampled** from that Gaussian. To keep backpropagation possible through the random sampling, the sample is written as $\pred{z} = \pred{\mu} + \pred{\sigma} \odot \epsilon$ with $\epsilon$ drawn from a standard normal: the **reparameterisation trick**. The loss adds a second term to the reconstruction loss, the **KL divergence** between each input's Gaussian and a standard normal, which pulls all codes toward one shared, gap-free region. After training, sampling $\pred{z}$ from a standard normal and decoding it produces new, plausible data.

<Formula name="VAE loss (negative ELBO)" tag="Generative autoencoder" words="Reconstruction loss plus a KL term that keeps each input's latent Gaussian close to a standard normal, so the latent space has no holes and can be sampled." remember="Rebuild well, stay close to normal.">

$$
\begin{aligned}
\pred{z} &= \pred{\mu} + \pred{\sigma} \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \\
\loss{L} &= \lVert \data{x} - \pred{\hat x} \rVert^2 + \hyper{\beta} \cdot \tfrac{1}{2} \textstyle\sum_j \big(\pred{\mu_j}^2 + \pred{\sigma_j}^2 - 1 - \log \pred{\sigma_j}^2\big)
\end{aligned}
$$

- $\pred{\mu}, \pred{\sigma}$: the encoder's mean and standard deviation per latent dimension
- the second term is $\mathrm{KL}(\mathcal{N}(\pred{\mu}, \pred{\sigma}^2) \,\Vert\, \mathcal{N}(0, 1))$; $\hyper{\beta} = 1$ is the standard VAE, $\hyper{\beta} \gt 1$ a β-VAE

**Example:** a dimension with $\pred{\mu} = 1, \pred{\sigma} = 1$ costs $\tfrac12(1 + 1 - 1 - 0) = 0.5$; one with $\pred{\mu} = 0, \pred{\sigma} = 0.5$ costs $\tfrac12(0.25 - 1 + 1.386) = 0.318$ (both checked with `torch.distributions.kl_divergence`).

</Formula>

:::tip[🎯 Interview angle]
**"Can you generate new images with an autoencoder?"** Not reliably with a plain one: its latent codes are scattered, and decoding a random point gives garbage. A VAE makes the encoder output a Gaussian per input, samples from it with the reparameterisation trick, and adds a KL term that pulls all codes toward a standard normal, so sampling from that normal and decoding produces new data. VAE samples tend to be blurrier than GAN samples.
:::

## 6. GANs: Two Networks in Competition

A **generative adversarial network (GAN)** learns to produce realistic samples by setting up a game between two networks. The **generator** $G$ takes a random noise vector $\data{z}$ and turns it into a sample $G(\data{z})$, a fake. It never sees real data directly. The **discriminator** $D$ is a binary classifier: given a sample, it outputs the probability that it is **real** (from the dataset) rather than fake (from $G$). The two are trained in alternation: $D$ learns to tell real from fake, and $G$ learns to produce fakes that $D$ classifies as real.

The only way the generator learns is through the discriminator's **gradient**. When $G$'s fake gets a low "real" probability, backpropagation through $D$ into $G$ tells $G$ how to change its output to look more real. As $D$ gets better at spotting fakes, $G$ is pushed to make better ones. In theory, the game ends when $G$ produces exactly the data distribution and $D$ can do no better than guessing, outputting 0.5 for everything. After training, the discriminator is thrown away and the generator is the product: feed it new noise, get new samples.

![GAN loop: random noise z goes into the generator, which outputs a fake; real data and fakes go into the discriminator, which outputs the probability of real; the discriminator's gradient flows back to the generator](/img/deep-learning/diagrams/gan-loop.svg)

*How to read it:* black arrows are the forward pass. The dashed red arrow is the only way information reaches the generator: the gradient of the discriminator's verdict.

<Formula name="GAN minimax objective" tag="The adversarial game" words="The discriminator maximises its log-likelihood of labelling real data real and fakes fake; the generator minimises the same quantity by making D call its fakes real." remember="D wants to catch, G wants to fool.">

$$
\begin{aligned}
\min_{\param{G}} \max_{\param{D}} \; & \mathbb{E}_{\data{x} \sim p_{data}} \big[\log \pred{D}(\data{x})\big] \\
+ \; & \mathbb{E}_{\data{z} \sim p_z} \big[\log\big(1 - \pred{D}(\param{G}(\data{z}))\big)\big]
\end{aligned}
$$

- $\pred{D}(\cdot)$: the discriminator's probability that its input is real
- $\param{G}(\data{z})$: a fake made from noise $\data{z}$ (2 random numbers here)
- in practice $G$ minimises $-\log \pred{D}(\param{G}(\data{z}))$ instead (the "non-saturating" loss), which gives stronger gradients early, when $D$ easily rejects every fake

**Example:** for a fixed $G$ the best discriminator is $D^*(x) = p_{data}(x) / (p_{data}(x) + p_g(x))$; if $G$ is perfect, $D^* = 0.5$ everywhere and the discriminator's loss (real + fake binary cross-entropy) is $2 \ln 2 = 1.386$.

</Formula>

:::analogy[Counterfeiter and bank teller]
A counterfeiter (generator) prints notes; a bank teller (discriminator) checks them. Each rejected note teaches the counterfeiter what gave it away, and each clever fake forces the teller to look closer. The counterfeiter never sees a real note up close; everything they know comes from the teller's reactions.
:::

:::tip[🎯 Interview angle]
**"What is the basic idea of a GAN?"** Two networks trained in competition: a generator turns random noise into fake samples, and a discriminator classifies samples as real or fake. The discriminator is trained to catch fakes and the generator is trained, through the discriminator's gradient, to fool it. At the ideal equilibrium the generator matches the data distribution and the discriminator outputs 0.5. Training is hard because the two must stay balanced.
:::

## 7. Training a GAN and What Balance Looks Like

Each training step has two halves. **Discriminator step:** take a batch of real samples and a batch of fakes from the current generator, and update $D$ with binary cross-entropy so that real → 1 and fake → 0. The fakes are **detached** from the generator's graph, because this step must not change $G$. **Generator step:** make a fresh batch of fakes, pass them through $D$, and update only $G$ so that $D$ would call them real (target 1, the non-saturating loss). Both use Adam, conventionally with $\beta_1 = 0.5$ for GANs.

The toy experiment makes the dynamics visible. The real data are 8 Gaussian blobs (the **modes**) on a circle of radius 2. Generator and discriminator are small MLPs (two hidden layers of 128), the noise has 2 dimensions, and both learn at $10^{-3}$. After 100 steps the generator's samples cover 2 of the 8 modes, after 250 steps 5, and from step 500 all 8; by step 3000, 91.8% of its samples lie within 0.25 of a real mode centre. The numbers at step 3000 also match the theory of a balanced game: the discriminator gives real points an average probability of 0.511 and fakes 0.457, both close to 0.5, and its loss is 1.343, close to the $2 \ln 2 = 1.386$ of a discriminator that can't tell the difference.

A GAN's losses do not steadily decrease the way a classifier's do, because each network's loss depends on the other. A discriminator loss near $2\ln 2$ is a sign of balance, not of failure; a discriminator loss near zero means it has won, and the generator stops getting useful gradients. That is why GANs are judged by looking at samples and by sample-quality metrics such as **FID** (Fréchet Inception Distance), not by their loss curves.

| Quantity at step 3000 | Balanced run | What a balanced game predicts |
|---|---|---|
| Mean $D$(real) | 0.511 | 0.5 |
| Mean $D$(fake) | 0.457 | 0.5 |
| Discriminator loss | 1.343 | $2 \ln 2$ = 1.386 |
| Modes covered | 8 of 8 | all 8 |
| Samples near a real mode | 91.8% | high |

:::tip[🎯 Interview angle]
**"How do you train a GAN, and how do you know it's working?"** Alternate a discriminator step (real → 1, detached fakes → 0) and a generator step (fakes → 1, update only G). Losses don't go to zero; a discriminator loss near $2\ln 2$ with $D$ around 0.5 on both real and fake means balance. Judge progress by samples and metrics like FID; in the toy run, the balanced GAN reached all 8 modes with $D$(real) 0.511 and $D$(fake) 0.457.
:::

## 8. Mode Collapse: Good-Looking but Not Diverse

**Mode collapse** happens when the generator produces only a few kinds of output instead of the full variety in the data. Trained on handwritten digits, a collapsed generator might draw convincing 1s and 7s and nothing else. Each sample can look perfectly realistic, which is what makes collapse easy to miss: the problem is **diversity**, not quality. It happens because the generator is rewarded only for fooling the current discriminator. If one kind of output fools it reliably, the cheapest strategy is to produce mostly that; when the discriminator adapts, the generator may hop to another mode instead of covering all of them.

The toy run reproduces it with one change: the discriminator's learning rate is raised to $10^{-2}$ and the generator's lowered to $10^{-4}$, so the discriminator learns 100 times faster. Everything else, including the random seed, is the same. This time the generator settles on **2 of the 8 modes** by step 500 and stays there through step 3000. Its samples look fine individually: 93.4% of them lie on a real mode, even more than the healthy run's 91.8%. But 6 of the 8 modes are never produced. The discriminator has clearly won: it rates real points 0.78 and fakes 0.19, and its loss is 0.577, far below $2 \ln 2$.

![Real data with 8 modes; healthy GAN at step 3000 covering all 8; mode-collapsed GAN at step 3000 covering only 2](/img/deep-learning/vision-generative/gan-ring.png)

*How to read it:* each panel plots 1,000 points. The collapsed generator (right) maps all its noise onto a short arc between two neighbouring modes.

<Animation src="generative/gan-mode-collapse" caption="Two GANs trained on the same 8-mode data, sampled at the same steps. With balanced learning rates the samples spread to all 8 green rings; with a 100x faster discriminator they lock onto 2 and stay there." />

| | Balanced run | Discriminator 100× faster |
|---|---|---|
| Learning rates (D / G) | $10^{-3}$ / $10^{-3}$ | $10^{-2}$ / $10^{-4}$ |
| Modes covered at step 3000 | 8 of 8 | 2 of 8 |
| Samples on a real mode | 91.8% | 93.4% |
| Mean $D$(real) / $D$(fake) | 0.511 / 0.457 | 0.78 / 0.19 |
| Discriminator loss | 1.343 | 0.577 |

Common remedies attack the incentive or the instability. **Minibatch discrimination** lets the discriminator compare samples within a batch, so a batch of near-identical fakes is easy to reject. **Wasserstein GAN (WGAN)** with a gradient penalty replaces the real/fake classifier with a critic whose score gives smooth, informative gradients even when the generator is far off. **Spectral normalisation** limits how sharply the discriminator can change, and the **two time-scale update rule** tunes the two learning rates separately, which is exactly the balance this experiment broke. **Unrolled GANs** let the generator anticipate the discriminator's next update.

:::danger[Interview Trap: "the samples look great, so the GAN works"]
Mode collapse produces samples that each look real (93.4% on a real mode here) while missing most of the data's variety. Check diversity too: count classes or modes in a large sample, or use metrics such as FID that compare the whole distribution.
:::

:::analogy[A comedian with one joke]
A comedian discovers one joke that always gets a laugh from tonight's audience and tells it all night. Every telling works, so by the audience's measure each joke is "good", but the act has no range. A GAN generator under mode collapse is that comedian; the discriminator is the audience it has learned to please.
:::

:::tip[🎯 Interview angle]
**"What is mode collapse, and why is it a problem?"** The generator produces only a few kinds of output (only some digits, say) instead of the full data distribution. Each sample can look realistic, so it is easy to miss, but the model hasn't learned the distribution. In a toy run, making the discriminator learn 100× faster collapsed the generator onto 2 of 8 modes. Remedies include balanced or separately tuned learning rates, minibatch discrimination, WGAN-GP and spectral normalisation; diversity metrics like FID detect it.
:::

## 9. Autoencoders, VAEs, GANs and Diffusion Compared

These families answer different questions. A plain **autoencoder** is a representation learner: it is good at compression, denoising and anomaly scores, and it isn't a generator. A **VAE** is a probabilistic autoencoder: its smooth latent space supports sampling and interpolation, with a principled likelihood-based loss, but its samples tend to be blurry because averaging over uncertainty smooths details. A **GAN** gives sharp, realistic samples but has no encoder, can't directly evaluate how likely a sample is, and is unstable to train (mode collapse). **Diffusion models**, behind most current image generators, learn to remove noise step by step; they train stably, cover modes well and produce high-quality samples, at the cost of many network evaluations per sample.

| | Autoencoder | VAE | GAN | Diffusion |
|---|---|---|---|---|
| Learns | Compact code to rebuild input | Smooth latent distribution | A sampler that fools a critic | To reverse a noising process |
| Loss | Reconstruction | Reconstruction + KL | Adversarial (minimax) | Noise-prediction MSE |
| Generates new data? | Not reliably | Yes (often blurry) | Yes (sharp) | Yes (sharp, diverse) |
| Has an encoder? | Yes | Yes | No | No (noise is the "latent") |
| Training stability | Easy | Easy | Hard (balance, mode collapse) | Easy, slow sampling |
| Typical uses | Compression, denoising, anomaly detection | Generation, interpolation, latent features | Image synthesis, super-resolution, style transfer | Text-to-image, high-quality synthesis |

:::tip[🎯 Interview angle]
**"Autoencoder vs VAE vs GAN?"** An autoencoder learns a compact code to reconstruct its input and is used for compression, denoising and anomaly detection, not generation. A VAE makes the code a smooth distribution so you can sample new data, with a reconstruction + KL loss, but samples are blurrier. A GAN trains a generator against a discriminator and gives sharp samples, but has no encoder and is prone to unstable training and mode collapse. Diffusion models now lead image generation by training stably on a denoising objective.
:::

## ⚙️ Going Deeper (Training Internals)

**Why `detach()` in the discriminator step.** The fakes are produced by $G$, so without `detach()` the discriminator's `backward()` would also compute gradients for the generator's weights. Those gradients would point the wrong way (helping $D$) and, with a shared optimizer or without zeroing, could leak into $G$'s update. Detaching also saves the cost of backpropagating through $G$.

**Saturating vs non-saturating generator loss.** Early in training $D$ rejects fakes confidently, $D(G(z)) \approx 0$, and $\log(1 - D(G(z)))$ is almost flat there, so minimising it gives the generator nearly zero gradient. Maximising $\log D(G(z))$ instead (implemented as BCE with target 1) has the same fixed point but steep gradients exactly where the generator is worst. All modern GAN code uses the non-saturating form.

**Linear autoencoder = PCA.** With linear layers and MSE, the optimal encoder spans the same subspace as the top $k$ principal components (though not necessarily the same basis vectors). The nonlinear autoencoder's advantage on digits (0.0145 vs 0.0246 at $k = 8$) is the benefit of the ReLU layers.

**Other regularised autoencoders.** Besides a narrow bottleneck, overcomplete autoencoders can be kept from copying by a **sparsity** penalty on the code (sparse autoencoders, now also used to interpret features inside large language models), by corrupting the input (denoising) or by penalising the encoder's sensitivity to the input (contractive autoencoders).

**Convolutional autoencoders.** For real images the encoder is a CNN (convolutions with stride or pooling) and the decoder uses **transposed convolutions** or upsampling, the same encoder-decoder shape as segmentation networks on the [previous page](./01-object-detection-and-segmentation.md).

**Choosing the anomaly threshold.** The threshold should come from **normal validation data** (a percentile of its errors) or, if some labelled anomalies exist, from the precision-recall trade-off the application can afford. Reconstruction error also depends on how "complex" an input is, so simple anomalies can be rebuilt well; on this page, many 9s were rebuilt about as well as normal digits.

**DCGAN rules of thumb.** For image GANs, the DCGAN guidelines remain the starting point: strided convolutions instead of pooling, batch normalisation in both networks (except G's output and D's input), ReLU in G with a tanh output, LeakyReLU in D, and Adam with learning rate $2 \times 10^{-4}$ and $\beta_1 = 0.5$.

## 💻 Python Implementation

:::tip[Run it yourself]
`pip install torch scikit-learn`. Everything runs on a CPU in under a minute per block; outputs are from a real run (PyTorch 2.15, scikit-learn 1.9.1).
:::

### 1. Load the 8 × 8 digits

1,797 images, each 64 pixels from 0 to 16, scaled to 0–1 and split 75/25.

```python title="autoencoder.py" showLineNumbers
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")                     # 1,797 images of 8 x 8 pixels, scaled to 0..1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(Xtr.shape, Xte.shape)                          # (1347, 64) (450, 64)
```

### 2. The autoencoder

Encoder 64 → 32 → k, decoder k → 32 → 64 with a sigmoid so outputs stay in 0–1.

```python showLineNumbers
class AE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, k))
        self.decoder = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)                          # the latent code
        return self.decoder(z), z

torch.manual_seed(0)
model = AE(k=8)
print(sum(p.numel() for p in model.parameters()))    # 4744
```

### 3. Train on reconstruction loss

The usual five-step loop; the target is the input itself (or a clean version of it, for denoising).

```python showLineNumbers
def train(model, X_in, X_target, epochs=300, lr=1e-2, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xi, Xt = torch.tensor(X_in), torch.tensor(X_target)
    for epoch in range(epochs):
        perm = torch.randperm(len(Xi))
        for i in range(0, len(Xi), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            x_hat, _ = model(Xi[b])
            # highlight-next-line
            loss = loss_fn(x_hat, Xt[b])             # reconstruction loss: compare with the target, no labels
            loss.backward()
            opt.step()
    return model

train(model, Xtr, Xtr)
with torch.no_grad():
    x_hat, z = model(torch.tensor(Xte))
print(round(nn.functional.mse_loss(x_hat, torch.tensor(Xte)).item(), 5), tuple(z.shape))   # 0.01447 (450, 8)
print(round(float(((Xte - Xtr.mean(0)) ** 2).mean()), 5))                                  # 0.07403  mean-image baseline
```

### 4. Compare with PCA

The linear baseline with the same 8 numbers per image.

```python showLineNumbers
from sklearn.decomposition import PCA

pca = PCA(8).fit(Xtr)
print(round(float(((pca.inverse_transform(pca.transform(Xte)) - Xte) ** 2).mean()), 5))   # 0.02461
```

### 5. Anomaly detection with reconstruction error

Train on digits 0–8 only; digit 9 is the anomaly.

```python showLineNumbers
from sklearn.metrics import roc_auc_score

normal = Xtr[ytr != 9]
detector = train(AE(8), normal, normal)
with torch.no_grad():
    # highlight-next-line
    err = ((detector(torch.tensor(Xte))[0] - torch.tensor(Xte)) ** 2).mean(1).numpy()   # one score per image
is_nine = (yte == 9)
print(round(float(err[~is_nine].mean()), 5), round(float(err[is_nine].mean()), 5), round(roc_auc_score(is_nine, err), 4))
# 0.0143 0.03086 0.8836
thr = np.percentile(err[~is_nine], 95)               # threshold from normal data only
print(round(float(thr), 5), int((err[is_nine] > thr).sum()), int((err[~is_nine] > thr).sum()))
# 0.03098 17 21     -> 17 of 45 nines caught, 21 of 405 normal digits flagged
```

### 6. Denoising autoencoder

Noisy input, clean target.

```python showLineNumbers
rng = np.random.default_rng(0)
noisy_tr = np.clip(Xtr + rng.normal(0, 0.3, Xtr.shape), 0, 1).astype("float32")
noisy_te = np.clip(Xte + rng.normal(0, 0.3, Xte.shape), 0, 1).astype("float32")
# highlight-next-line
denoiser = train(AE(16), noisy_tr, Xtr)              # input: noisy, target: clean
with torch.no_grad():
    clean = denoiser(torch.tensor(noisy_te))[0].numpy()
print(round(float(((noisy_te - Xte) ** 2).mean()), 5), round(float(((clean - Xte) ** 2).mean()), 5))   # 0.05237 0.02374
```

### 7. A GAN on 8 Gaussian blobs, healthy and collapsed

The same code with two learning-rate settings; the D step uses detached fakes, the G step uses the non-saturating loss.

```python title="gan_ring.py" showLineNumbers
import math, numpy as np, torch, torch.nn as nn

def ring(n, rng, r=2.0, std=0.05):                  # 8 Gaussian blobs on a circle of radius 2
    k = rng.integers(0, 8, n)
    a = 2 * math.pi * k / 8
    return np.stack([r * np.cos(a), r * np.sin(a)], 1) + rng.normal(0, std, (n, 2))

def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))

def train_gan(lr_g=1e-3, lr_d=1e-3, steps=3000, bs=256, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    G, D = mlp(2, 2), mlp(2, 1)                      # generator: noise z (2 numbers) -> point; discriminator: point -> logit
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ones, zeros = torch.ones(bs, 1), torch.zeros(bs, 1)
    for step in range(steps):
        real = torch.tensor(ring(bs, rng), dtype=torch.float)
        # highlight-next-line
        fake = G(torch.randn(bs, 2)).detach()        # detach: the D step must not update G
        opt_d.zero_grad()
        d_loss = bce(D(real), ones) + bce(D(fake), zeros)
        d_loss.backward(); opt_d.step()
        opt_g.zero_grad()
        # highlight-next-line
        g_loss = bce(D(G(torch.randn(bs, 2))), ones)   # non-saturating: G wants D to say "real"
        g_loss.backward(); opt_g.step()
    return G, D, d_loss.item(), g_loss.item()

CENTRES = np.array([[2 * math.cos(2 * math.pi * k / 8), 2 * math.sin(2 * math.pi * k / 8)] for k in range(8)])

def modes_covered(points, tol=0.25):
    d = np.linalg.norm(points[:, None] - CENTRES[None], axis=2)
    near = d.min(1) < tol
    counts = np.bincount(d.argmin(1)[near], minlength=8)
    return int((counts >= 0.01 * len(points)).sum()), round(float(near.mean()), 3)

z = torch.randn(1000, 2, generator=torch.Generator().manual_seed(123))
real_eval = torch.tensor(ring(1000, np.random.default_rng(999)), dtype=torch.float)
for name, cfg in [("healthy", {}), ("collapsed", {"lr_d": 1e-2, "lr_g": 1e-4})]:
    G, D, dl, gl = train_gan(**cfg)
    with torch.no_grad():
        pts = G(z)
        print(name, round(dl, 4), round(gl, 4), round(torch.sigmoid(D(real_eval)).mean().item(), 4),
              round(torch.sigmoid(D(pts)).mean().item(), 4), modes_covered(pts.numpy()))
# healthy 1.3426 0.8696 0.5114 0.4565 (8, 0.918)       D loss near 2 ln 2, D near 0.5: balanced
# collapsed 0.5769 1.9365 0.7815 0.1878 (2, 0.934)     D wins; 2 of 8 modes, yet 93% of samples look "real"
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"What is the difference between an autoencoder and a GAN?"** An autoencoder is trained to reconstruct its input through a bottleneck with a reconstruction loss; it learns a compact representation (compression, denoising, anomaly detection) and isn't a reliable generator. A GAN trains a generator against a discriminator; it learns to produce new realistic samples from noise but has no encoder and is unstable to train. For generation from an autoencoder-like model, use a VAE.
:::

- **Autoencoders need no labels**: the input is the target.
- **No bottleneck, no learning**: a code as large as the input lets the network copy (test MSE 0.0057 at 64, but no summary).
- **Lower reconstruction error isn't the goal**: an autoencoder that rebuilds everything is useless for anomaly detection.
- **A linear autoencoder is PCA**; the nonlinear one beat PCA at every size here.
- **Plain autoencoders can't generate**: random latent points decode to noise; use a VAE.
- **Anomaly thresholds come from normal data** and always trade recall against false alarms (17/45 caught at 21/405 false alarms).
- **The generator never sees real data**; it learns only through the discriminator's gradient.
- **Detach the fakes in the discriminator step.**
- **Use the non-saturating generator loss**; the original $\log(1 - D(G(z)))$ gives vanishing gradients early.
- **GAN losses don't decrease like a classifier's**; a D loss near $2\ln 2 = 1.386$ means balance.
- **Mode collapse hides behind good samples**: 93.4% of collapsed samples were on a real mode.
- **Balance matters**: a 100× faster discriminator collapsed the generator onto 2 of 8 modes.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Autoencoders**
- Encoder $\pred{z} = f_{\param{\theta}}(\data{x})$, decoder $\pred{\hat x} = g_{\param{\phi}}(\pred{z})$; trained to reconstruct the input, no labels.
- Reconstruction loss: MSE $\frac1d \lVert \data{x} - \pred{\hat x} \rVert^2$ (or BCE for 0–1 data).
- Bottleneck stops copying; digits test MSE 0.0358 / 0.0145 / 0.0076 for k = 2 / 8 / 16; mean image 0.0740; PCA(8) 0.0246.
- Latent space clusters similar inputs without labels; linear AE = PCA.

**Uses**
- Denoising: noisy in, clean target (MSE 0.0524 → 0.0237).
- Anomaly detection: train on normal, score by reconstruction error, threshold from normal data (AUC 0.8836; 17/45 caught at 5.2% false alarms).
- VAE: Gaussian latent, reparameterisation $\pred{z} = \pred{\mu} + \pred{\sigma} \odot \epsilon$, loss = reconstruction + KL; can sample.

**GANs**
- Generator: noise → fake. Discriminator: sample → P(real). Trained in alternation, competition.
- $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$; non-saturating G loss in practice.
- Balance: $D$ ≈ 0.5 on both, D loss ≈ $2\ln 2$ = 1.386 (toy: 0.511, 0.457, 1.343).
- Mode collapse: limited variety; toy run with a 100× faster D: 2 of 8 modes, 93.4% on a mode, D loss 0.577.
- Remedies: balanced/TTUR learning rates, minibatch discrimination, WGAN-GP, spectral norm; measure diversity (FID).

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Encoder / decoder | $\pred{z} = f_{\param{\theta}}(\data{x})$, $\ \pred{\hat x} = g_{\param{\phi}}(\pred{z})$ | Squeeze to a code, rebuild from it |
| Reconstruction loss | $\loss{L} = \frac{1}{d} \sum_i (\data{x_i} - \pred{\hat x_i})^2$ | Average squared pixel error |
| Anomaly score | $s(\data{x}) = \lVert \data{x} - \pred{\hat x} \rVert^2 / d$; anomaly if $s \gt \hyper{\tau}$ | Badly rebuilt means unusual |
| Reparameterisation | $\pred{z} = \pred{\mu} + \pred{\sigma} \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ | Sample in a differentiable way |
| VAE KL term | $\tfrac12 \sum_j (\pred{\mu_j}^2 + \pred{\sigma_j}^2 - 1 - \log \pred{\sigma_j}^2)$ | Keep codes near a standard normal |
| GAN objective | $\min_{\param{G}} \max_{\param{D}} \mathbb{E}[\log \pred{D}(\data{x})] + \mathbb{E}[\log(1 - \pred{D}(\param{G}(\data{z})))]$ | D catches fakes, G fools D |
| Non-saturating G loss | $\loss{L_G} = -\mathbb{E}[\log \pred{D}(\param{G}(\data{z}))]$ | Strong gradients when G is bad |
| Optimal discriminator | $D^*(x) = \dfrac{p_{data}(x)}{p_{data}(x) + p_g(x)}$ | 0.5 everywhere when G is perfect |
| Balanced D loss | $2 \ln 2 = 1.386$ | D can't tell real from fake |

## Concept-Check Answers (from the question bank)

These are the revision question bank's autoencoder and GAN questions (Phase 5, Q44–Q47), answered in full; the bank page has all 51.

<details>
<summary><strong>Q44. What is an autoencoder? What are its two main parts?</strong></summary>

A neural network trained to reconstruct its input. Its two parts are the **encoder**, which compresses the input into a lower-dimensional latent representation, and the **decoder**, which reconstructs the input from that representation. The goal is to learn useful compressed representations; the **bottleneck** stops the model from simply copying its input and forces it to learn compact features (Chapters 1 and 3).

</details>

<details>
<summary><strong>Q45. What is reconstruction loss? How is it used in autoencoders?</strong></summary>

It measures the difference between the input $\data{x}$ and the reconstruction $\pred{\hat x}$, commonly mean squared error $\lVert \data{x} - \pred{\hat x} \rVert^2$ (averaged over pixels). The autoencoder is trained to minimise it, and a low value means a good reconstruction. Trained on normal data only, the autoencoder can then flag anomalies by their high reconstruction error: on the digits, unseen 9s had about twice the error of normal digits (AUC 0.88) (Chapters 2 and 4).

</details>

<details>
<summary><strong>Q46. What is the basic idea of a GAN? What are the generator and discriminator?</strong></summary>

A GAN has two networks trained in competition. The **generator** turns random noise into fake samples that should look real; the **discriminator** tries to tell real samples from fakes. The discriminator is trained to catch fakes, and the generator improves by learning, through the discriminator's gradient, to fool it. Training is difficult because the two must stay balanced: in a toy run, balanced learning rates gave $D \approx 0.5$ on real and fake and all 8 modes learned (Chapters 6 and 7).

</details>

<details>
<summary><strong>Q47. What is mode collapse in GANs? Why is it a problem?</strong></summary>

The generator produces only a limited variety of outputs instead of the full diversity of the data; trained on handwritten digits, it might repeatedly produce only one or two digit types. It's a problem because the generator hasn't learned the data distribution, and it is easy to miss because each sample can look realistic. In a toy run with a 100× faster discriminator, the generator covered 2 of 8 modes while 93.4% of its samples looked real. Mode collapse is one reason GAN training is considered unstable (Chapter 8).

</details>
