import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

// Layer rules from ARCHITECTURE.md §4, enforced with no-restricted-imports only.
const deny = (...groups) => ({ 'no-restricted-imports': ['error', { patterns: groups.map((g) => ({ group: [g], message: 'layer rule (ARCHITECTURE §4)' })) }] });

export default tseslint.config(
  { ignores: ['dist', 'external', 'docs', 'weights', 'tools', 'dev-dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: { globals: { ...globals.browser, ...globals.worker } },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
  { files: ['src/app/**'], rules: deny('**/engine/**', '**/models/**', '**/formats/**', '**/dsp/**') },
  // UI: components read the store through useApp/selectors and change it only through controller actions (review R2 S-1).
  {
    files: ['src/app/components/**'],
    rules: {
      'no-restricted-imports': ['error', {
        patterns: ['**/engine/**', '**/models/**', '**/formats/**', '**/dsp/**'].map((g) => ({ group: [g], message: 'layer rule (ARCHITECTURE §4)' })),
        paths: ['../store', '../../store', '@/app/store'].map((name) => ({ name, importNames: ['set', 'get'], message: 'write the store through a controller action' })),
      }],
    },
  },
  { files: ['src/audio/**'], rules: deny('**/engine/**', '**/models/**', '**/formats/**', '**/app/**') },
  { files: ['src/engine/**', 'src/models/**'], rules: deny('**/app/**', '**/audio/**', 'react', 'react-dom') },
  { files: ['src/formats/**', 'src/dsp/**'], rules: deny('**/app/**', '**/audio/**', '**/engine/**', '**/models/**', 'onnxruntime-web*', 'react') },
  { files: ['src/shared/**'], rules: deny('**/app/**', '**/audio/**', '**/engine/**', '**/models/**', '**/formats/**', '**/dsp/**') },
  // Rule 2: no DOM below the app; this code runs in the worker and in Node tests (review R3 A8).
  {
    files: ['src/{engine,models,formats,dsp,shared}/**'],
    rules: {
      'no-restricted-globals': ['error', ...['document', 'window', 'localStorage', 'sessionStorage'].map((name) => ({ name, message: 'no DOM in the engine layers (ARCHITECTURE §4)' }))],
    },
  },
);
