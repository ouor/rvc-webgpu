// (3) Options card. OptionsFields is reused by the result screen's rerun panel.
import { useState } from 'react';
import { ChevronDown, RotateCcw } from 'lucide-react';
import { Accordion as A } from 'radix-ui';
import { SectionLabel, SliderField, StepCard } from './Field';
import { fmtTime, parseTime } from '../lib/format';
import { resetOptions, setOpts, updateSettings } from '../controller';
import { DEFAULT_OPTIONS, useApp } from '../store';
import { useT } from '../i18n';

function RangeInputs() {
  const t = useT();
  const duration = useApp((s) => s.song?.duration);
  const range = useApp((s) => s.opts.range);
  const [da, setA] = useState<string>(), [db, setB] = useState<string>();
  const a = da ?? fmtTime(range?.start ?? 0), b = db ?? fmtTime(range?.end ?? duration ?? 0);
  const commit = () => {
    setA(undefined); setB(undefined);
    if (!duration) return;
    let s = parseTime(a), e = parseTime(b);
    if (!Number.isFinite(s)) s = range?.start ?? 0;
    if (!Number.isFinite(e)) e = range?.end ?? duration;
    s = Math.max(0, Math.min(s, duration)); e = Math.max(0, Math.min(e, duration));
    if (e - s < 1) return;
    setOpts({ range: s <= 0 && e >= duration - 0.05 ? undefined : { start: s, end: e } });
  };
  const box = 'w-20 h-8 rounded-md border border-input bg-card px-2 text-center tnum focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:bg-muted disabled:opacity-45';
  return (
    <div>
      <div className="flex items-center gap-2 text-small">
        <input aria-label={t('opt.rangeStart')} className={box} value={a} disabled={!duration} onChange={(e) => setA(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()} />
        <span aria-hidden>–</span>
        <input aria-label={t('opt.rangeEnd')} className={box} value={b} disabled={!duration} onChange={(e) => setB(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()} />
        {range && <button type="button" className="text-caption font-semibold text-accent-foreground" onClick={() => setOpts({ range: undefined })}>{t('opt.rangeAll')}</button>}
      </div>
      <p className="text-caption text-muted-foreground mt-1.5">{duration ? t('opt.rangeHint') : t('opt.rangeNoSong')}</p>
    </div>
  );
}

const ADV_KEYS = ['indexRate', 'protect', 'rmsMixRate', 'sid', 'aggressiveness'] as const;
/** Fine-tune values that differ from the defaults (only the rows that are shown count). */
function useAdvChanged(rerun?: boolean) {
  const opts = useApp((s) => s.opts);
  const hasIndex = useApp((s) => s.index?.status === 'ready');
  const f0 = useApp((s) => s.model?.info?.f0 ?? true);
  return ADV_KEYS.some((k) => opts[k] !== DEFAULT_OPTIONS[k]
    && (k !== 'indexRate' || hasIndex) && (k !== 'protect' || f0) && (k !== 'aggressiveness' || opts.mode === 'song'))
    || (!rerun && !!opts.range);
}

/** DESIGN 9.6: pitch on the card, everything else under "세부 조정". `rerun` hides the range (a range change means a new separation). */
export function OptionsFields({ rerun }: { rerun?: boolean }) {
  const t = useT();
  const opts = useApp((s) => s.opts);
  const info = useApp((s) => (s.model?.status === 'ready' ? s.model.info : undefined));
  const hasIndex = useApp((s) => s.index?.status === 'ready');
  const advancedOpen = useApp((s) => s.settings.advancedOpen);
  const advChanged = useAdvChanged(rerun);
  const song = opts.mode === 'song';
  const nof0 = info && !info.f0;
  const keyWarn = song && opts.key % 12 !== 0 && !nof0;
  const maxSid = Math.max(0, (info?.nSpk ?? 1) - 1);

  return (
    <div className="space-y-5">
      <SliderField
        label={t('opt.key')} name={t('opt.key')} help={t('opt.keyHelp')} value={opts.key} onChange={(key) => setOpts({ key })}
        badge={song && <span className="text-caption text-muted-foreground ml-1">{t('opt.keyKeep')}</span>}
        min={-24} max={24} step={1} unit={t('opt.semitone')} defaultValue={0}
        disabled={nof0} note={nof0 ? t('opt.nof0') : undefined} warning={keyWarn ? t('opt.keyWarn') : undefined}
        chips={[{ label: t('opt.chipDown'), value: -12 }, { label: t('opt.chipZero'), value: 0 }, { label: t('opt.chipUp'), value: 12 }]}
        valueText={(v) => t('opt.keyText', { n: v > 0 ? `+${v}` : v })}
      />
      <A.Root type="single" collapsible value={advancedOpen ? 'adv' : ''} onValueChange={(v) => updateSettings({ advancedOpen: v === 'adv' })}>
        <A.Item value="adv" id="advanced" className="rounded-xl bg-muted/60 data-[state=open]:bg-transparent data-[state=open]:ring-1 data-[state=open]:ring-border">
          <A.Header>
            <A.Trigger className="group w-full h-11 px-4 flex items-center gap-2 text-small font-semibold rounded-xl focus-visible:outline-2 focus-visible:outline-ring">
              <span className="flex-1 text-left">{t('opt.advanced')}</span>
              {advChanged && !advancedOpen && <span className="inline-flex items-center gap-1.5 text-caption text-accent-foreground"><span className="size-1.5 rounded-full bg-primary" />{t('opt.changedDot')}</span>}
              <ChevronDown className="size-4 transition-transform duration-200 group-data-[state=open]:rotate-180" />
            </A.Trigger>
          </A.Header>
          <A.Content className="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
            <div className="px-4 pb-4 space-y-5">
              {hasIndex && <SliderField label={t('opt.indexRate')} help={t('opt.indexHelp')} value={opts.indexRate} onChange={(indexRate) => setOpts({ indexRate })}
                min={0} max={1} step={0.05} defaultValue={DEFAULT_OPTIONS.indexRate} />}
              {song && <SliderField label={t('opt.agg')} help={t('opt.aggHelp')} value={opts.aggressiveness} onChange={(aggressiveness) => setOpts({ aggressiveness })}
                min={0} max={20} step={1} defaultValue={DEFAULT_OPTIONS.aggressiveness} note={rerun ? t('opt.aggRerun') : undefined} />}
              {!nof0 && <SliderField label={t('opt.protect')} help={t('opt.protectHelp')} value={opts.protect} onChange={(protect) => setOpts({ protect })}
                min={0} max={0.5} step={0.01} defaultValue={DEFAULT_OPTIONS.protect} note={t('opt.protectNote')} />}
              <SliderField label={t('opt.rms')} help={t('opt.rmsHelp')} value={opts.rmsMixRate} onChange={(rmsMixRate) => setOpts({ rmsMixRate })}
                min={0} max={1} step={0.05} defaultValue={DEFAULT_OPTIONS.rmsMixRate} />
              <SidInput value={opts.sid} max={maxSid} />
              {!rerun && (
                <div className="pt-4 border-t">
                  <SectionLabel>{t('opt.secRange')}</SectionLabel>
                  <RangeInputs />
                </div>
              )}
              <button type="button" className="inline-flex items-center gap-1 text-small font-semibold text-muted-foreground hover:text-foreground" onClick={() => resetOptions(rerun)}>
                <RotateCcw className="size-3.5" />{t('opt.resetAll')}
              </button>
            </div>
          </A.Content>
        </A.Item>
      </A.Root>
    </div>
  );
}

/** Speaker number: always shown (the checkpoint's speaker count is not reliable enough to hide it), clamped to the embedding table. */
function SidInput({ value, max }: { value: number; max: number }) {
  const t = useT();
  const [draft, setDraft] = useState<string>();
  const commit = () => {
    if (draft === undefined) return;
    const n = Math.round(Number(draft));
    setDraft(undefined);
    if (Number.isFinite(n)) setOpts({ sid: Math.max(0, Math.min(max, n)) });
  };
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <label htmlFor="sid" className="text-small font-semibold">{t('opt.sid')}</label>
        <p className="text-caption text-muted-foreground mt-0.5">{t('opt.sidNote')}</p>
      </div>
      <input id="sid" type="number" inputMode="numeric" min={0} max={max} value={draft ?? value}
        onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()}
        className="w-16 h-8 rounded-md border border-input bg-card text-right px-2 text-small tnum shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" />
    </div>
  );
}

export function OptionsCard() {
  const t = useT();
  const opts = useApp((s) => s.opts);
  const adv = useAdvChanged();
  const changed = opts.key !== DEFAULT_OPTIONS.key || adv;
  return (
    <StepCard n={3} id="c3" title={t('card.options')} aside={changed
      ? <button type="button" className="text-caption font-semibold text-accent-foreground hover:underline underline-offset-2" onClick={() => resetOptions()}>{t('opt.changed')}</button>
      : <span className="text-caption text-muted-foreground">{t('opt.defaultsOk')}</span>}>
      <OptionsFields />
    </StepCard>
  );
}
