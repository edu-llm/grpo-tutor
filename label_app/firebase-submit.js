/* Firestore submission for the labelling app.
 *
 * Exposed as `window.submitLabel(label)` so app.js stays a plain script with no
 * build step. Submission is best-effort by design: every label is already in
 * localStorage before this is called, and the download button is the fallback,
 * so a failed write costs nothing but a retry.
 *
 * Anonymous sign-in is attempted but not required. The provider is a console
 * toggle; if it is off the write still goes through as an unauthenticated
 * create, which the rules permit deliberately (volunteers cannot be asked to
 * register). Turning it on tightens the rules without touching this file.
 */
import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js';
import {
  getFirestore, collection, addDoc, serverTimestamp,
} from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js';
import {
  getAuth, signInAnonymously,
} from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js';

const firebaseConfig = {
  apiKey: 'AIzaSyA0OdNn-p5WPGkqGc_7C-wD0tAKmmfcxh4',
  authDomain: 'grpo-tutor-label.firebaseapp.com',
  projectId: 'grpo-tutor-label',
  storageBucket: 'grpo-tutor-label.firebasestorage.app',
  messagingSenderId: '366024689289',
  appId: '1:366024689289:web:e8f12915ecf1ee76f7e858',
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

// fire and forget; `uid` is simply absent if the provider is disabled
let uid = null;
signInAnonymously(auth)
  .then((cred) => { uid = cred.user.uid; })
  .catch(() => { /* provider off - unauthenticated writes are allowed */ });

window.submitLabel = async function submitLabel(label, source) {
  const doc = {
    who: String(label.who).slice(0, 40),
    source: String(source).slice(0, 120),
    itemId: String(label.id).slice(0, 60),
    kind: label.kind,
    createdAt: serverTimestamp(),
  };
  if (uid) doc.uid = uid;
  if (label.kind === 'turn') {
    doc.leak = Number(label.leak);          // 1-3, higher = more of the answer given away
    doc.goodness = Number(label.goodness);  // 1-5, higher = better teaching
  } else { doc.winner = label.winner; }
  if (label.note) doc.note = String(label.note).slice(0, 600);

  await addDoc(collection(db, 'labels'), doc);
};

window.SUBMIT_READY = true;
