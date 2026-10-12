/**
 * <Animation src="cnn/convolution" caption="..." />
 *
 * Plays a Manim-rendered clip from static/animations/<src>.mp4 like a silent,
 * looping GIF, with study controls: play/pause, restart, and 0.5× slow motion.
 * - Only plays while on screen (saves CPU on long pages).
 * - Respects "reduce motion": starts paused on the poster frame.
 * Registered globally in src/theme/MDXComponents.js, so pages need no import.
 */
import React, {useEffect, useRef, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

export default function Animation({src, caption}) {
  const webmUrl = useBaseUrl(`/animations/${src}.webm`);
  const mp4Url = useBaseUrl(`/animations/${src}.mp4`);
  const posterUrl = useBaseUrl(`/animations/${src}.jpg`);
  const ref = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [slow, setSlow] = useState(false);
  const userPaused = useRef(false);

  // Auto-play when visible, pause when scrolled away (unless reduced motion).
  useEffect(() => {
    const video = ref.current;
    if (!video) return undefined;
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) userPaused.current = true;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !userPaused.current) {
          video.play().catch(() => {});
        } else if (!entry.isIntersecting) {
          video.pause();
        }
      },
      {threshold: 0.35},
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (ref.current) ref.current.playbackRate = slow ? 0.5 : 1;
  }, [slow]);

  const toggle = () => {
    const video = ref.current;
    if (!video) return;
    if (video.paused) {
      userPaused.current = false;
      video.play().catch(() => {});
    } else {
      userPaused.current = true;
      video.pause();
    }
  };

  const restart = () => {
    const video = ref.current;
    if (!video) return;
    video.currentTime = 0;
    userPaused.current = false;
    video.play().catch(() => {});
  };

  return (
    <figure className={styles.figure}>
      <div className={styles.frame}>
        <video
          ref={ref}
          className={styles.video}
          poster={posterUrl}
          muted
          loop
          playsInline
          preload="metadata"
          onClick={toggle}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          aria-label={caption || 'Animation'}>
          {/* Browsers pick the first format they can play: H.264 MP4 almost
              everywhere; VP9 WebM for builds without H.264 (e.g. open-source Chromium) */}
          <source src={mp4Url} type="video/mp4" />
          <source src={webmUrl} type="video/webm" />
        </video>
        <div className={styles.controls}>
          <button type="button" onClick={toggle} title={playing ? 'Pause' : 'Play'}>
            {playing ? '❚❚' : '▶'}
          </button>
          <button type="button" onClick={restart} title="Restart">
            ↺
          </button>
          <button
            type="button"
            onClick={() => setSlow((s) => !s)}
            title="Toggle slow motion"
            className={slow ? styles.active : undefined}>
            0.5×
          </button>
        </div>
      </div>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}
    </figure>
  );
}
