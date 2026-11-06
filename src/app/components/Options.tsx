// (3) Options card. OptionsFields is reused by the result screen's rerun panel.
import { useState } from 'react';
import { ChevronDown, RotateCcw } from 'lucide-react';
import { Accordion as A } from 'radix-ui';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { SectionLabel, SliderField, StepCard } from './Field';
import { fmtTime, parseTime } from '../lib/format';
import { setOpts, updateSettings } from '../controller';
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

/** Basic + advanced option fields. `rerun` hides the input-range section (a range change means a new separation). */
export function OptionsFields({ rerun }: { rerun?: boolean }) {
  const t = useT();
  const opts = useApp((s) => s.opts);
  const info = useApp((s) => (s.model?.status === 'ready' ? s.model.info : undefined));
  const hasIndex = useApp((s) => s.index?.status === 'ready');
  const advancedOpen = useApp((s) => s.settings.advancedOpen);
  const song = opts.mode === 'song';
  const nof0 = info && !info.f0;
  const keyWarn = song && opts.key % 12 !== 0 && !nof0;
  const multiSpk = !!info && info.nSpk > 1;
  const advCount = 2 + (multiSpk ? 1 : 0) + (song ? 1 : 0) + (rerun ? 0 : 1);

  return (
    <div className="space-y-5">
      <SliderField
        label={t('opt.key')} help={t('opt.keyHelp')} value={opts.key} onChange={(key) => setOpts({ key })}
        min={-24} max={24} step={1} unit={t('opt.semitone')} defaultValue={0}
        disabled={nof0} note={nof0 ? t('opt.nof0') : undefined} warning={keyWarn ? t('opt.keyWarn') : undefined}
        chips={[{ label: t('opt.chipDown'), value: -12 }, { label: '0', value: 0 }, { label: t('opt.chipUp'), value: 12 }]}
        valueText={(v) => t('opt.keyText', { n: v > 0 ? `+${v}` : v })}
      />
      <SliderField
        label={t('opt.indexRate')} help={t('opt.indexHelp')} value={hasIndex ? opts.indexRate : 0} onChange={(indexRate) => setOpts({ indexRate })}
        min={0} max={1} step={0.05} defaultValue={DEFAULT_OPTIONS.indexRate} disabled={!hasIndex} note={!hasIndex ? t('opt.indexOff') : undefined}
      />
      <A.Root type="single" collapsible value={advancedOpen ? 'adv' : ''} onValueChange={(v) => updateSettings({ advancedOpen: v === 'adv' })}>
        <A.Item value="adv" id="advanced" className="rounded-xl bg-muted/60 data-[state=open]:bg-transparent data-[state=open]:ring-1 data-[state=open]:ring-border">
          <A.Header>
            <A.Trigger className="group w-full h-11 px-4 flex items-center justify-between text-small font-semibold rounded-xl focus-visible:outline-2 focus-visible:outline-ring">
              <span>{t('opt.advanced')} <span className="font-normal text-muted-foreground">· {t('opt.count', { n: advCount })}</span></span>
              <ChevronDown className="size-4 transition-transform duration-200 group-data-[state=open]:rotate-180" />
            </A.Trigger>
          </A.Header>
          <A.Content className="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
            <div className="px-4 pb-4 space-y-5">
              <div>
                <SectionLabel>{t('opt.secConvert')}</SectionLabel>
                <div className="space-y-4">
                  {multiSpk && (
                    <div>
                      <label className="text-small font-semibold mb-2 block" id="sid-l">{t('opt.sid')}</label>
                      <Select value={String(opts.sid)} onValueChange={(v) => setOpts({ sid: Number(v) })}>
                        <SelectTrigger aria-labelledby="sid-l" className="w-full h-10 rounded-lg"><SelectValue /></SelectTrigger>
                        <SelectContent>{Array.from({ length: info.nSpk }, (_, i) => <SelectItem key={i} value={String(i)}>{t('opt.sidItem', { n: i })}</SelectItem>)}</SelectContent>
                      </Select>
                      <p className="text-caption text-muted-foreground mt-1.5">{t('opt.sidNote')}</p>
                    </div>
                  )}
                  <SliderField label={t('opt.protect')} help={t('opt.protectHelp')} value={opts.protect} onChange={(protect) => setOpts({ protect })}
                    min={0} max={0.5} step={0.01} defaultValue={DEFAULT_OPTIONS.protect} disabled={nof0} note={nof0 ? t('opt.nof0') : t('opt.protectNote')} />
                  <SliderField label={t('opt.rms')} help={t('opt.rmsHelp')} value={opts.rmsMixRate} onChange={(rmsMixRate) => setOpts({ rmsMixRate })}
                    min={0} max={1} step={0.05} defaultValue={DEFAULT_OPTIONS.rmsMixRate} />
                  <div className="flex items-center justify-between text-small"><span className="text-muted-foreground">{t('opt.f0Method')}</span><span className="font-semibold">RMVPE</span></div>
                </div>
              </div>
              {song && (
                <div className="pt-4 border-t">
                  <SectionLabel>{t('opt.secSeparate')}</SectionLabel>
                  <div className="flex items-center justify-between text-small mb-4"><span className="text-muted-foreground">{t('opt.sepModel')}</span><span className="font-semibold">UVR VR · uvr-vocal</span></div>
                  <SliderField label={t('opt.agg')} help={t('opt.aggHelp')} value={opts.aggressiveness} onChange={(aggressiveness) => setOpts({ aggressiveness })}
                    min={0} max={20} step={1} defaultValue={DEFAULT_OPTIONS.aggressiveness} note={rerun ? t('opt.aggRerun') : undefined} />
                </div>
              )}
              {!rerun && (
                <div className="pt-4 border-t">
                  <SectionLabel>{t('opt.secRange')}</SectionLabel>
                  <RangeInputs />
                </div>
              )}
              <button type="button" className="inline-flex items-center gap-1 text-small font-semibold text-muted-foreground hover:text-foreground"
                onClick={() => setOpts({ ...DEFAULT_OPTIONS, mode: opts.mode, range: rerun ? opts.range : undefined })}>
                <RotateCcw className="size-3.5" />{t('opt.resetAll')}
              </button>
            </div>
          </A.Content>
        </A.Item>
      </A.Root>
    </div>
  );
}

export function OptionsCard() {
  const t = useT();
  return (
    <StepCard n={3} id="c3" title={t('card.options')} aside={<span className="text-caption text-muted-foreground">{t('opt.defaultsOk')}</span>}>
      <OptionsFields />
    </StepCard>
  );
}
