# Ghid scurt pentru sustinere

Acest fisier este pentru explicarea proiectului la prezentare. Nu este documentatia oficiala, ci o fisa de orientare rapida.

## Ideea proiectului

Proiectul construieste un agent care joaca un joc de tip Block Blast. La fiecare pas agentul vede tabla, cele trei piese disponibile si pozitiile unde pot fi puse. Apoi alege piesa si pozitia. Scopul este sa supravietuiasca multe etape si sa elimine cat mai multe linii.

Am folosit invatare prin recompensa, mai exact PPO cu action masking. Action masking inseamna ca modelul nu poate alege mutari imposibile, cum ar fi plasarea unei piese peste alte piese sau in afara tablei.

## Fisiere principale

`blocks.py`

Contine toate formele. Fiecare forma este o matrice cu valori 0 si 1. Tot aici sunt categoriile de forme: simple, medii, grele, mici, mari etc.

`engine.py`

Este motorul jocului. Aici sunt regulile: tabla, verificarea mutarilor, plasarea pieselor, stergerea liniilor si generarea mainii urmatoare. Nu contine retele neuronale.

`env.py`

Transforma motorul intr-un mediu compatibil Gymnasium. Aici sunt observatiile, actiunile, action mask-ul, reward-ul si CNN-urile folosite de model.

`train.py`

Porneste antrenarea PPO. Creeaza mai multe medii in paralel, configureaza modelul, scrie metrici in TensorBoard si salveaza modelul.

`watch.py`

Este interfata vizuala. De aici pot testa un model, pot urmari euristica, pot juca eu contra AI-ului si pot construi manual o tabla pe care modelul incearca sa o rezolve.

`behavioral_cloning.py`

Este experimentul in care euristica este folosita ca profesor. Se colecteaza mutari bune de la euristica, apoi se incearca pre-antrenarea modelului prin invatare supervizata.

## Workflow tehnic

1. `blocks.py` da lista de piese.
2. `engine.py` creeaza un joc si genereaza trei piese.
3. `env.py` transforma starea jocului in observatie pentru model.
4. `train.py` trimite observatia catre PPO.
5. PPO alege o actiune valida.
6. `env.py` aplica actiunea in motor si calculeaza recompensa.
7. Experienta este folosita pentru update-ul modelului.
8. TensorBoard arata daca modelul invata sau nu.
9. `watch.py` permite verificarea comportamentului vizual.

## Observatia primita de model

Modelul primeste un dictionar cu:

- `board`: tabla curenta;
- `valid_actions`: unde poate fi pusa fiecare piesa;
- `hand`: cele trei piese disponibile;
- `available`: ce piese mai sunt disponibile in mana curenta.

Pentru 8x8 exista 192 actiuni posibile: 3 piese x 64 pozitii. Pentru 4x4 exista 48 actiuni.

## Reward-uri

Am incercat mai multe variante:

- reward pentru linii eliminate;
- reward pentru terminarea unei etape;
- penalizare pentru final de joc;
- penalizare pentru gauri;
- reward pentru contact.

Reward-ul de contact a fost important. Ideea este ca o piesa pusa langa marginea tablei sau langa alte piese este de obicei mai buna decat o piesa aruncata izolata in mijlocul tablei.

## Generatorul de piese

Generatorul a fost o parte foarte importanta. Daca piesele sunt prea grele, modelul pierde imediat si nu invata. Daca piesele sunt prea usoare, modelul invata un joc artificial.

Varianta principala foloseste un generator adaptiv:

- mai multe piese simple;
- mai putine piese grele;
- piese mari mai des cand tabla este goala;
- piese mari foarte rar cand tabla este aproape plina.

## CNN si PPO

Am folosit CNN deoarece tabla este o structura spatiala. O retea complet conectata simpla pierde relatia dintre celule vecine.

CNN-ul proceseaza tabla si hartile de actiuni valide. Apoi informatia despre piese este combinata cu partea spatiala. PPO foloseste aceasta reprezentare pentru doua lucruri:

- actorul alege actiunea;
- criticul estimeaza cat de buna este starea.

## De ce action masking

Fara action masking, modelul ar putea alege multe actiuni imposibile. Asta ar irosi mult timp de antrenare. Cu action masking, actiunile invalide sunt scoase inainte de alegerea actiunii.

## Euristica

Euristica verifica toate mutarile valide si le da un scor. Ea tine cont de:

- cate linii se sterg;
- cat contact are piesa;
- cate mutari raman dupa plasare;
- cat de compacta ramane tabla;
- cat de mare este cea mai mare zona libera.

Euristica a ajuns la peste 100 de etape in unele teste, deci generatorul permite jocuri lungi. Asta a aratat ca problema principala era modelul si procesul de invatare, nu imposibilitatea jocului.

## Experimentul 4x4

Am testat si o tabla 4x4 ca sa am FPS mai mare si spatiu de actiuni mai mic. Rezultatele au fost mai slabe decat pe 8x8. Explicatia este ca tabla este mult mai aglomerata, iar o piesa mare poate bloca jocul imediat.

## Intrebari probabile

De ce PPO?

Pentru ca este stabil, functioneaza bine cu politici stocastice si se potriveste cu action masking prin `MaskablePPO`.

De ce nu Q-learning tabelar?

Pentru ca numarul de stari este urias. Numai tabla 8x8 are foarte multe configuratii posibile, iar piesele din mana cresc si mai mult spatiul.

De ce CNN?

Pentru ca jocul depinde de geometrie: linii, coloane, margini, contact, goluri si zone libere.

Ce a mers cel mai bine?

Un model 8x8 cu generator adaptiv, reward de contact si reward pentru linii. Nu a ajuns la nivelul euristicii, dar a fost cea mai coerenta directie.

Ce nu a mers?

Penalizarea pentru gauri nu a fost foarte clara pentru model. De asemenea, behavioral cloning a invatat bine pe train, dar a generalizat slab pe validare.

Ce as imbunatati mai departe?

As folosi mai multe date de la euristica, o arhitectura mai buna pentru fiecare actiune si o combinatie intre pre-antrenare supervizata si PPO.

## Cum explic folosirea LLM

LLM-ul a fost folosit ca ajutor pentru:

- interfata vizuala din `watch.py`;
- verificarea unor inconsistente intre fisiere;
- curatarea si organizarea codului;
- completari punctuale de text/cod pornind de la structura deja aleasa.

Partea importanta este ca rezultatele au fost validate prin rulare locala, TensorBoard si testare vizuala.
