// Panel "Scenario editor": a form for curated fields of a scenario YAML and a YAML editor, a live
// preview (loader validation, scenario key, changes) and "Save as new scenario…" into the user
// scenario directory. The base file is never changed; saving writes a new file (or replaces an own one).

const PREVIEW_DELAY_MS = 350;
const TABS = [{ value: 'form', label: 'Form', icon: 'sliders' }, { value: 'yaml', label: 'YAML', icon: 'code' }];

function esc(text) {
  return String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Same idea as pylovo-ui's config editor: keys, strings, numbers, booleans and comments per line.
function highlightYaml(text) {
  return text.split('\n').map((line) => {
    let quote = null;
    let commentAt = -1;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if ((c === '"' || c === "'") && quote === null) quote = c;
      else if (c === quote) quote = null;
      else if (c === '#' && quote === null && (i === 0 || /\s/.test(line[i - 1]))) { commentAt = i; break; }
    }
    const body = commentAt >= 0 ? line.slice(0, commentAt) : line;
    const comment = commentAt >= 0 ? line.slice(commentAt) : '';
    const value = (v) => esc(v)
      .replace(/(&quot;.*?&quot;|&#39;.*?&#39;)/g, '<span class="y-string">$1</span>')
      .replace(/\b(true|false|True|False|null)\b/g, '<span class="y-bool">$1</span>')
      .replace(/(^|[\s[,{:])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?=[\s,\]}]|$)/g, '$1<span class="y-number">$2</span>');
    const m = /^(\s*)(- )?([A-Za-z_][\w]*|\d+)(:)(.*)$/.exec(body);
    const html = m
      ? `${m[1]}${m[2] || ''}<span class="y-key">${esc(m[3])}</span>:${value(m[5])}`
      : value(body);
    return html + (comment ? `<span class="y-comment">${esc(comment)}</span>` : '');
  }).join('\n') + '\n';
}

// Equal YAML values for the changed markers (numbers by value).
function same(a, b) {
  if (a === undefined) a = null;
  if (b === undefined) b = null;
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  return JSON.stringify(a) === JSON.stringify(b);
}

const round = (x, digits) => Number(Number(x).toFixed(digits));

export function createScenariosPanel(ctx) {
  const { host, state } = ctx;
  return {
    name: 'GridExpandScenarios',
    data() {
      return {
        TABS, tab: 'form', baseName: null, form: null, loading: false, loadError: null,
        changes: {}, text: null, raw: {},
        preview: null, previewing: false, previewError: null, showChanges: false,
        dialog: null, saving: false, deleting: false,
      };
    },
    computed: {
      s() { return state; },
      options() {
        const own = state.scenarios.filter((x) => x.user);
        const shipped = state.scenarios.filter((x) => !x.user);
        return { own, shipped };
      },
      fields() { return (this.form?.editable_fields || []).flatMap((sec) => sec.fields); },
      fieldByKey() { return Object.fromEntries(this.fields.map((f) => [f.key, f])); },
      baseValues() { return Object.fromEntries(this.fields.map((f) => [f.key, f.present || f.optional ? f.value : null])); },
      dirty() { return this.text !== null || Object.keys(this.changes).length > 0; },
      yamlText: {
        get() { return this.text ?? this.form?.text ?? ''; },
        set(value) { this.text = value; },
      },
      highlighted() { return highlightYaml(this.yamlText); },
      gutter() { return Array.from({ length: this.yamlText.split('\n').length }, (_, i) => i + 1); },
      issues() { return this.preview?.issues || []; },
      errors() { return this.issues.filter((i) => i.level === 'error'); },
      errorLines() { return new Set(this.issues.filter((i) => i.line && i.level === 'error').map((i) => i.line)); },
      valueChanges() { return (this.preview?.changes || []).filter((c) => c.key !== 'scenario.id'); },
      changeCount() { return this.preview ? this.valueChanges.length : Object.keys(this.changes).length; },
      proposal() { return this.form?.proposal || { file_name: null, scenario_id: null }; },
      canSave() { return !!this.form && this.dirty && !this.previewing && !this.errors.length && this.form.user_dir?.writable; },
      saveTitle() {
        if (!this.form?.user_dir?.writable) return this.form?.user_dir?.reason || 'The user scenario directory is not writable';
        if (!this.dirty) return 'Change a value first';
        if (this.errors.length) return 'Fix the errors first';
        return 'Save the edited copy as a new file in your scenario directory';
      },
      statusBadge() {
        if (this.previewing) return { cls: 'accent', text: 'checking…' };
        if (this.previewError) return { cls: 'bad', text: 'preview failed' };
        if (!this.preview) return { cls: '', text: '–' };
        if (this.errors.length) return { cls: 'bad', text: `${this.errors.length} error(s)` };
        const warnings = this.issues.length - this.errors.length;
        return warnings ? { cls: 'warn', text: `valid · ${warnings} warning(s)` } : { cls: 'good', text: 'valid' };
      },
      dialogPreview() { return this.dialog?.preview || null; },
      dialogTarget() { return this.dialogPreview?.target || null; },
      dialogErrors() { return (this.dialogPreview?.issues || []).filter((i) => i.level === 'error'); },
      dialogCanSave() {
        const d = this.dialog;
        if (!d || this.saving || d.loading || !this.dialogPreview || this.dialogErrors.length) return false;
        if (!this.dialogTarget?.allowed) return false;
        return !this.dialogTarget.exists || d.replace;
      },
    },
    watch: {
      's.scenarios': { immediate: true, handler() { if (!this.baseName && !state.editScenario) this.pickDefault(); } },
      's.editScenario': { immediate: true, handler(name) { if (name) this.openRequested(name); } },
      changes: { deep: true, handler() { this.schedulePreview(); } },
      text() { this.schedulePreview(); },
      'dialog.fileName'() { this.scheduleDialogPreview(); },
      'dialog.scenarioId'() { this.scheduleDialogPreview(); },
    },
    mounted() {
      ctx.ensureLoaded();
      this.onKeyDown = (e) => { if (this.dialog && e.key === 'Escape') this.closeDialog(); };
      window.addEventListener('keydown', this.onKeyDown);
    },
    beforeUnmount() {
      window.removeEventListener('keydown', this.onKeyDown);
      clearTimeout(this.previewTimer);
      clearTimeout(this.dialogTimer);
    },
    methods: {
      enc: encodeURIComponent,
      pickDefault() {
        const list = state.scenarios.filter((x) => x.valid && !x.template);
        const synthetic = list.find((x) => Object.values(x.adoption || {}).every((a) => a.mode === 'deterministic_share'));
        const pick = synthetic || list[0] || state.scenarios[0];
        if (pick) this.load(pick.name);
      },
      async confirmDiscard() {
        if (!this.dirty) return true;
        return host.ui.confirmDialog({
          title: 'Discard your edits?', confirmText: 'Discard edits', danger: true,
          body: `The unsaved changes to the copy of ${this.baseName} are lost.`,
        });
      },
      async openRequested(name) {
        state.editScenario = null;
        if (name === this.baseName && !this.dirty) return;
        if (await this.confirmDiscard()) this.load(name);
      },
      async pickBase(event) {
        const name = event.target.value;
        if (name === this.baseName) return;
        if (await this.confirmDiscard()) this.load(name);
        else event.target.value = this.baseName;
      },
      reset() {
        this.changes = {};
        this.text = null;
        this.raw = {};
        this.preview = null;
        this.previewError = null;
      },
      async load(name) {
        this.baseName = name;
        this.loading = true;
        this.reset();
        try {
          const form = await host.api(`scenarios/${this.enc(name)}/form`);
          if (this.baseName !== name) return;
          this.form = form;
          this.loadError = null;
          if (!form.editable_fields.length) this.tab = 'yaml';
          this.schedulePreview(0);
        } catch (err) {
          this.form = null;
          this.loadError = err.message;
        } finally {
          this.loading = false;
        }
      },
      reload() {
        ctx.loadScenarios();
        if (this.baseName) this.confirmDiscard().then((ok) => ok && this.load(this.baseName));
      },

      // ------------------------------------------------------------- form values
      isPercent(f) { return f.type === 'percent'; },
      valueOf(key) {
        if (key in this.changes) return this.changes[key];
        if (this.text !== null && this.preview?.values && key in this.preview.values) return this.preview.values[key];
        return this.baseValues[key];
      },
      inactive(f) { return !!f.requires && this.valueOf(f.requires.key) !== f.requires.value; },
      baseInactive(f) { return !!f.requires && this.baseValues[f.requires.key] !== f.requires.value; },
      isChanged(f) {
        if (this.inactive(f) !== this.baseInactive(f)) return true;
        return !this.inactive(f) && !same(this.valueOf(f.key), this.baseValues[f.key]);
      },
      display(f) {
        if (this.raw[f.key] !== undefined) return this.raw[f.key];
        if (this.inactive(f)) return '';
        const v = this.valueOf(f.key);
        if (v === null || v === undefined) return '';
        return this.isPercent(f) ? String(round(v * 100, 8)) : String(v);
      },
      scale(f, x) { return x === undefined || x === null ? undefined : this.isPercent(f) ? round(x * 100, 8) : x; },
      setValue(f, value) {
        if (this.text === null && same(value, this.baseValues[f.key])) delete this.changes[f.key];
        else this.changes[f.key] = value;
      },
      onNumber(f, event) {
        const text = event.target.value;
        this.raw[f.key] = text;
        if (text.trim() === '') return;
        let v = Number(text.replace(',', '.'));
        if (!Number.isFinite(v)) return;
        if (this.isPercent(f)) v = round(v / 100, 10);
        if (f.type === 'int' && !Number.isInteger(v)) return;
        this.setValue(f, v);
      },
      onBlur(f) { delete this.raw[f.key]; },
      onEnum(f, event) {
        const option = f.options.find((o) => String(o.value) === event.target.value);
        if (option) this.setValue(f, option.value);
      },
      resetField(f) {
        delete this.raw[f.key];
        const controlling = f.requires && this.fieldByKey[f.requires.key];
        if (controlling && this.inactive(f) !== this.baseInactive(f)) this.resetField(controlling);
        const base = this.baseValues[f.key];
        if (this.text !== null && base !== null && base !== undefined) this.changes[f.key] = base;
        else delete this.changes[f.key];
      },
      fieldIssue(f) { return this.issues.find((i) => i.key === f.key) || null; },
      revert() { this.reset(); this.schedulePreview(0); },

      // ------------------------------------------------------------- preview
      body(extra = {}) {
        const body = { base: this.baseName, changes: { ...this.changes }, base_sha256: this.form?.text_sha256,
          scenario_id: this.proposal.scenario_id, file_name: this.proposal.file_name, ...extra };
        if (this.text !== null) body.text = this.text;
        if (!body.scenario_id) delete body.scenario_id;
        if (!body.file_name) delete body.file_name;
        return body;
      },
      schedulePreview(ms = PREVIEW_DELAY_MS) {
        clearTimeout(this.previewTimer);
        if (!this.form) return;
        this.previewing = true;
        this.previewTimer = setTimeout(() => this.runPreview(), ms);
      },
      async runPreview() {
        const seq = (this.previewSeq = (this.previewSeq || 0) + 1);
        const base = this.baseName;
        this.previewing = true;
        try {
          const res = await host.api('scenarios/preview', { method: 'POST', body: this.body() });
          if (seq !== this.previewSeq || base !== this.baseName) return;
          this.preview = res;
          this.previewError = null;
        } catch (err) {
          if (seq !== this.previewSeq) return;
          this.previewError = err.message;
        } finally {
          if (seq === this.previewSeq) this.previewing = false;
        }
      },
      async flushPreview() {
        if (!this.previewing) return;
        clearTimeout(this.previewTimer);
        await this.runPreview();
      },
      async setTab(tab) {
        if (tab === this.tab) return;
        if (tab === 'yaml' && Object.keys(this.changes).length) {
          // The YAML view edits one text: apply the form changes to it first.
          await this.flushPreview();
          if (this.preview && !this.previewError) {
            this.text = this.preview.text;
            this.changes = {};
          }
        }
        this.raw = {};
        this.tab = tab;
      },
      jump(issue) {
        if (issue.key && this.fieldByKey[issue.key] && this.tab === 'form') {
          const el = this.$el.querySelector(`[data-key="${issue.key}"]`);
          el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
          el?.querySelector('input, select')?.focus({ preventScroll: true });
          return;
        }
        if (!issue.line) return;
        this.setTab('yaml').then(() => this.$nextTick(() => {
          const ta = this.$refs.ta;
          if (!ta) return;
          const lines = this.yamlText.split('\n');
          const pos = lines.slice(0, issue.line - 1).reduce((a, l) => a + l.length + 1, 0);
          ta.focus();
          ta.setSelectionRange(pos, pos + (lines[issue.line - 1] || '').length);
          this.$refs.editor.scrollTop = Math.max(0, (issue.line - 6) * 19);
        }));
      },
      onKey(e) {
        if (e.key === 'Tab') {
          e.preventDefault();
          const t = e.target;
          const start = t.selectionStart;
          this.text = this.yamlText.slice(0, start) + '  ' + this.yamlText.slice(t.selectionEnd);
          this.$nextTick(() => { t.selectionStart = t.selectionEnd = start + 2; });
        } else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
          e.preventDefault();
          if (this.canSave) this.openDialog();
        }
      },
      fmt(value, key) {
        if (value === null || value === undefined) return '–';
        const f = this.fieldByKey[key];
        if (f && this.isPercent(f) && typeof value === 'number') return `${round(value * 100, 6)} %`;
        if (f?.type === 'enum') return f.options.find((o) => o.value === value)?.label || String(value);
        if (typeof value === 'object') return JSON.stringify(value);
        return f?.unit && typeof value === 'number' ? `${value} ${f.unit}` : String(value);
      },
      changeLabel(c) { return c.label || c.key; },

      // ------------------------------------------------------------- save / delete
      async openDialog() {
        await this.flushPreview();
        if (!this.canSave) return;
        this.dialog = { fileName: this.proposal.file_name || '', scenarioId: this.proposal.scenario_id || '', replace: false,
          preview: null, loading: true, error: null };
        this.scheduleDialogPreview(0);
        this.$nextTick(() => this.$refs.dialogFile?.focus());
      },
      closeDialog() { if (!this.saving) this.dialog = null; },
      scheduleDialogPreview(ms = 300) {
        if (!this.dialog) return;
        clearTimeout(this.dialogTimer);
        this.dialog.loading = true;
        this.dialogTimer = setTimeout(() => this.runDialogPreview(), ms);
      },
      async runDialogPreview() {
        const d = this.dialog;
        if (!d) return;
        const seq = (this.dialogSeq = (this.dialogSeq || 0) + 1);
        try {
          const res = await host.api('scenarios/preview', { method: 'POST',
            body: this.body({ scenario_id: d.scenarioId.trim(), file_name: d.fileName.trim() }) });
          if (seq !== this.dialogSeq || d !== this.dialog) return;
          d.preview = res;
          d.error = null;
        } catch (err) {
          if (seq === this.dialogSeq && d === this.dialog) d.error = err.message;
        } finally {
          if (seq === this.dialogSeq && d === this.dialog) d.loading = false;
        }
      },
      async save() {
        const d = this.dialog;
        if (!this.dialogCanSave) return;
        this.saving = true;
        try {
          const entry = await host.api('scenarios', { method: 'POST',
            body: this.body({ scenario_id: d.scenarioId.trim(), file_name: d.fileName.trim(), overwrite: !!d.replace }) });
          this.dialog = null;
          host.ui.toast('good', 'Scenario saved', `${entry.name} · ${entry.scenario_key}${entry.backup ? ` · backup ${entry.backup}` : ''}`, 6000);
          await ctx.loadScenarios();
          state.focusScenario = entry.name;
          this.reset();
          await this.load(entry.name);
        } catch (err) {
          d.error = err.message;
          if (err.detail?.issues) d.preview = { ...(d.preview || {}), issues: err.detail.issues };
        } finally {
          this.saving = false;
        }
      },
      async remove() {
        const name = this.baseName;
        const ok = await host.ui.confirmDialog({
          title: `Delete ${name}?`, danger: true, confirmText: 'Delete scenario',
          body: `The file is renamed to ${name}.bak-<time> in your scenario directory (not erased) and leaves the list. `
            + 'Results already in the database stay.',
        });
        if (!ok) return;
        this.deleting = true;
        try {
          await host.api(`scenarios/${this.enc(name)}`, { method: 'DELETE' });
          host.ui.toast('good', 'Scenario deleted', name, 4000);
          this.reset();
          this.baseName = null;
          this.form = null;
          await ctx.loadScenarios();
          this.pickDefault();
        } catch (err) {
          host.ui.errorToast('Could not delete the scenario', err);
        } finally {
          this.deleting = false;
        }
      },
      diffClass(line) {
        if (line.startsWith('+++') || line.startsWith('---')) return 'meta';
        if (line.startsWith('+')) return 'add';
        if (line.startsWith('-')) return 'del';
        return line.startsWith('@@') ? 'hunk' : '';
      },
    },
    template: `
    <div class="panel gx-panel gx-scen">
      <div class="panel-toolbar nowrap">
        <div class="panel-title"><Icon name="sliders" :size="15"/>Scenario editor</div>
        <select class="select sm" :value="baseName" @change="pickBase" aria-label="Base scenario" title="Base scenario: the editor works on a copy">
          <optgroup v-if="options.own.length" label="Your scenarios">
            <option v-for="x in options.own" :key="x.name" :value="x.name">{{ x.name }}{{ x.valid ? '' : ' (invalid)' }}</option>
          </optgroup>
          <optgroup label="Shipped scenarios">
            <option v-for="x in options.shipped" :key="x.name" :value="x.name">{{ x.name }}{{ x.template ? ' (template)' : '' }}{{ x.valid ? '' : ' (invalid)' }}</option>
          </optgroup>
        </select>
        <span v-if="form" class="badge" :class="form.user ? 'accent' : ''" :title="form.path">{{ form.user ? 'yours' : 'shipped · read-only' }}</span>
        <span class="grow"></span>
        <Seg :modelValue="tab" @update:modelValue="setTab" :options="TABS"/>
        <button v-if="form && form.writable" class="btn ghost icon sm" title="Delete this scenario file (kept as a backup)" :disabled="deleting" @click="remove"><Icon name="trash" :size="14"/></button>
        <button class="btn ghost icon sm" title="Reload from disk" @click="reload"><Icon name="refresh" :size="14"/></button>
      </div>

      <EmptyState v-if="loadError" error title="Scenario not readable" :text="loadError"/>
      <EmptyState v-else-if="!form" loading title="Loading scenario"/>
      <template v-else>
        <div class="gx-summary">
          <div class="gx-keys">
            <div class="small muted ellipsis" :title="form.path">Base <span class="mono">{{ form.valid ? form.scenario_key : form.name }}</span></div>
            <div class="row" style="gap: 6px; min-width: 0">
              <span class="small muted">New</span>
              <span class="mono strong ellipsis" :title="'Scenario key of the copy saved as ' + (proposal.file_name || '…') + ' (results are keyed by it)'">{{ preview && preview.scenario_key ? preview.scenario_key : '…' }}</span>
              <span class="badge" :class="statusBadge.cls">{{ statusBadge.text }}</span>
            </div>
          </div>
          <button class="btn ghost sm" :disabled="!changeCount" @click="showChanges = !showChanges" :aria-expanded="showChanges">
            <Icon name="list" :size="13"/>{{ changeCount }} change(s)
          </button>
        </div>
        <div v-if="showChanges && valueChanges.length" class="gx-changes">
          <div v-for="c in valueChanges" :key="c.key" class="gx-change">
            <span class="ellipsis" :title="c.key">{{ changeLabel(c) }}</span>
            <span class="mono small"><span class="muted">{{ c.kind === 'added' ? 'new' : fmt(c.old, c.key) }}</span> → <span class="strong">{{ c.kind === 'removed' ? 'removed' : fmt(c.new, c.key) }}</span></span>
          </div>
        </div>
        <div v-if="issues.length || previewError || form.error" class="gx-issues stack tight">
          <div v-if="previewError" class="callout bad"><Icon class="ico" name="alert"/><div>{{ previewError }}</div></div>
          <div v-else-if="form.error && !preview" class="callout bad"><Icon class="ico" name="alert"/><div>{{ form.error }}</div></div>
          <div v-for="(i, n) in issues" :key="n" class="callout" :class="i.level === 'error' ? 'bad' : 'warn'" :style="i.line || i.key ? 'cursor: pointer' : ''" @click="jump(i)">
            <Icon class="ico" name="alert"/><div class="grow"><strong v-if="i.line">Line {{ i.line }}: </strong>{{ i.message }}</div></div>
        </div>

        <div v-if="tab === 'form'" class="panel-body stack loose">
          <div class="callout info"><Icon class="ico" name="info"/><div>Key assumptions of <span class="mono">{{ form.name }}</span>; help texts are the comments of the YAML file. The base file stays unchanged: <strong>Save as new scenario</strong> writes an edited copy (with its own scenario id and key) to your scenario directory. Everything else is edited in the <a href="#" @click.prevent="setTab('yaml')">YAML view</a>.</div></div>
          <div v-if="!form.editable_fields.length" class="callout warn"><Icon class="ico" name="alert"/><div>This file cannot be shown as a form; fix it in the YAML view.</div></div>
          <div v-for="sec in form.editable_fields" :key="sec.id" class="form-section">
            <h4>{{ sec.title }}</h4>
            <div v-if="sec.note" class="ff-help" style="margin: -4px 0 10px">{{ sec.note }}</div>
            <div class="form-grid">
              <div v-for="f in sec.fields" :key="f.key" :data-key="f.key" class="form-field" :class="{changed: isChanged(f), 'gx-bad': fieldIssue(f) && fieldIssue(f).level === 'error'}">
                <div class="ff-label" :title="f.hint"><span class="grow ellipsis">{{ f.label }}</span>
                  <button v-if="isChanged(f)" class="btn ghost icon sm gx-reset" :title="'Reset to the base value (' + fmt(baseValues[f.key], f.key) + ')'" @click="resetField(f)"><Icon name="undo" :size="12"/></button>
                </div>
                <Toggle v-if="f.type === 'bool'" :modelValue="!!valueOf(f.key)" @update:modelValue="(v) => setValue(f, v)" :label="valueOf(f.key) ? 'on' : 'off'"/>
                <select v-else-if="f.type === 'enum'" class="select" :value="String(valueOf(f.key))" @change="onEnum(f, $event)">
                  <option v-for="o in f.options" :key="String(o.value)" :value="String(o.value)">{{ o.label }}</option>
                </select>
                <div v-else class="input-unit">
                  <input class="input" type="number" :step="scale(f, f.step) || 'any'" :min="scale(f, f.min)" :max="scale(f, f.max)"
                         :value="display(f)" :disabled="inactive(f)" :placeholder="inactive(f) ? 'not used' : ''" :aria-label="f.label"
                         @input="onNumber(f, $event)" @blur="onBlur(f)">
                  <span v-if="f.unit" class="unit">{{ f.unit }}</span>
                </div>
                <div class="ff-key">{{ f.key }}</div>
                <div v-if="fieldIssue(f)" class="ff-help" :style="{color: fieldIssue(f).level === 'error' ? 'var(--critical-text)' : 'var(--warn-text)'}">{{ fieldIssue(f).message }}</div>
                <div v-if="inactive(f)" class="ff-help">Only used with deterministic_share.</div>
                <div class="ff-help">{{ f.help || f.hint }}</div>
              </div>
            </div>
          </div>
        </div>
        <div v-else class="code-editor" ref="editor">
          <div class="code-gutter"><div v-for="n in gutter" :key="n" :class="{err: errorLines.has(n)}">{{ n }}</div></div>
          <div class="code-area">
            <pre aria-hidden="true" v-html="highlighted"></pre>
            <textarea ref="ta" v-model="yamlText" wrap="off" spellcheck="false" autocapitalize="off" autocomplete="off" @keydown="onKey" :aria-label="'YAML of the copy of ' + form.name"></textarea>
          </div>
        </div>

        <div class="panel-toolbar" style="border-top: 1px solid var(--border); border-bottom: 0">
          <span class="small" :class="dirty ? 'strong' : 'muted'">{{ dirty ? (text !== null ? 'Edited YAML' : changeCount + ' unsaved change(s)') : 'No changes' }}</span>
          <span v-if="tab === 'yaml'" class="kbd">Ctrl S</span>
          <span class="grow"></span>
          <button class="btn ghost" :disabled="!dirty" @click="revert"><Icon name="undo" :size="13"/>Revert</button>
          <button class="btn primary" :disabled="!canSave" :title="saveTitle" @click="openDialog"><Icon name="save" :size="13"/>Save as new scenario…</button>
        </div>
      </template>

      <div v-if="dialog" class="modal-back" @mousedown.self="closeDialog">
        <div class="modal wide" role="dialog" aria-modal="true" aria-label="Save as new scenario">
          <div class="modal-head">
            <div class="modal-icon"><Icon name="save" :size="18"/></div>
            <div class="grow"><h3>Save as new scenario</h3><div class="small muted">Edited copy of {{ baseName }} → {{ form.user_dir.path }}</div></div>
          </div>
          <div class="modal-body">
            <div class="form-grid gx-dialog-grid">
              <div class="form-field">
                <label class="ff-label" for="gx-save-file">File name</label>
                <input id="gx-save-file" ref="dialogFile" class="input mono" v-model="dialog.fileName" spellcheck="false" autocomplete="off" @keydown.enter="save">
              </div>
              <div class="form-field">
                <label class="ff-label" for="gx-save-id">Scenario id</label>
                <input id="gx-save-id" class="input mono" v-model="dialog.scenarioId" spellcheck="false" autocomplete="off" @keydown.enter="save">
              </div>
            </div>
            <div class="card sunken gx-keyline">
              <div class="small muted">Scenario key of the new file (names its results)</div>
              <div class="row" style="gap: 8px"><span class="mono strong">{{ dialogPreview && dialogPreview.scenario_key ? dialogPreview.scenario_key : '…' }}</span><Spinner v-if="dialog.loading" :size="12"/></div>
            </div>
            <div v-if="dialog.error" class="callout bad"><Icon class="ico" name="alert"/><div>{{ dialog.error }}</div></div>
            <div v-if="dialogTarget && dialogTarget.message" class="callout" :class="dialogTarget.allowed ? 'warn' : 'bad'"><Icon class="ico" name="alert"/>
              <div class="grow">{{ dialogTarget.message }}
                <div v-if="dialogTarget.allowed && dialogTarget.exists" style="margin-top: 6px"><Toggle v-model="dialog.replace" label="Replace my file"/></div>
              </div>
            </div>
            <div v-for="(i, n) in (dialogPreview ? dialogPreview.issues : [])" :key="'i' + n" class="callout" :class="i.level === 'error' ? 'bad' : 'warn'"><Icon class="ico" name="alert"/><div>{{ i.message }}</div></div>
            <div v-if="dialogPreview && dialogPreview.changes.length" class="gx-changes boxed">
              <div v-for="c in dialogPreview.changes" :key="c.key" class="gx-change">
                <span class="ellipsis" :title="c.key">{{ changeLabel(c) }}</span>
                <span class="mono small"><span class="muted">{{ c.kind === 'added' ? 'new' : fmt(c.old, c.key) }}</span> → <span class="strong">{{ c.kind === 'removed' ? 'removed' : fmt(c.new, c.key) }}</span></span>
              </div>
            </div>
            <div v-if="dialogPreview && dialogPreview.diff" class="diff"><div v-for="(l, n) in dialogPreview.diff.split('\\n')" :key="n" :class="diffClass(l)">{{ l || ' ' }}</div></div>
          </div>
          <div class="modal-foot">
            <button class="btn" @click="closeDialog" :disabled="saving">Cancel</button>
            <button class="btn primary" :disabled="!dialogCanSave" @click="save"><Spinner v-if="saving" :size="12"/><Icon v-else name="save" :size="13"/>{{ dialogTarget && dialogTarget.exists ? 'Replace scenario' : 'Save scenario' }}</button>
          </div>
        </div>
      </div>
    </div>`,
  };
}
