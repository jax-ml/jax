// Helpers /////////////////////////////////////////////////////////////////////

// Returns a random float between min and max.
function rand(min, max) {
  return Math.random() * (max - min) + min;
}

// Formats the provided time as hh:mm:ss.
function formatTime(date) {
  // https://stackoverflow.com/a/25279399
  return date.toISOString().substring(11, 19);
}

// Periodically runs f with a delay between min_delay and max_delay.
// setIntervalWithJitter returns a cancel function that, when called, cancels
// the interval.
function setIntervalWithJitter(f, min_delay, max_delay) {
  let handle = null;

  f();
  const helper = () => {
    const g = () => {
      f();
      helper();
    };
    handle = setTimeout(g, rand(min_delay, max_delay));
    return () => {
      clearTimeout(handle);
    };
  };

  return helper();
}

// Coordination Service ////////////////////////////////////////////////////////

class CoordinationService {
  constructor(network, options) {
    const now = new Date();
    this.network = network;
    this.options = options;
    this.heartbeats = [now, now, now];
    this.alive = [true, true, true];
    this.in_barrier = [];

    // Periodically refresh state.
    setInterval(() => this.refresh(), 100);
  }

  receive(msg) {
    const {src, dst, type, payload} = msg;
    switch (type) {
      case 'heartbeat':
        this.heartbeats[src] = new Date();
        return [];
      case 'live_devices':
        if (this.options.barrier) {
          if (!this.in_barrier.includes(src)) {
            this.in_barrier.push(src);
            this.refresh_live_devices();
          }
        } else {
          this.network.push({
            src: 'server',
            dst: msg.src,
            type: 'live_devices',
            payload: this.live_devices(),
          })
        }
        break;
      default:
        console.log(`Unknown message type ${type}`)
    }
  }

  time_since_heartbeat(i) {
    return (new Date() - this.heartbeats[i]) / 1000;
  }

  detect_failures() {
    let something_failed = false;
    for (let i = 0; i < 3; ++i) {
      if (this.time_since_heartbeat(i) > 6) {
        if (this.alive[i]) {
          something_failed = true;
        }
        this.alive[i] = false;
      } else {
        this.alive[i] = true;
      }
    }

    if (something_failed && this.options.share_fate) {
      for (let i = 0; i < 3; ++i) {
        if (this.alive[i]) {
          this.network.push({
            src: 'server',
            dst: i,
            type: 'fail',
            payload: '💀',
          })
        }
      }
    }
  }

  live_devices() {
    let devices = [];
    for (let i = 0; i < 3; ++i) {
      if (this.alive[i]) {
        devices.push(i);
      }
    }
    return devices;
  }

  refresh_live_devices() {
    // Check dst see if the live_devices barrier is done.
    for (let i = 0; i < 3; ++i) {
      if (this.alive[i] && !this.in_barrier.includes(i)) {
        // The barrier isn't done.
        return;
      }
    }

    // The barrier is done! Send the set of live devices dst all live devices.
    let live = this.live_devices();
    for (let i of live) {
      this.network.push({
        src: 'server',
        dst: i,
        type: 'live_devices',
        payload: live,
      })
    }
    this.in_barrier = [];
  }

  refresh() {
    this.detect_failures();
    this.refresh_live_devices();
  }

  update_html(container) {
    for (let i = 0; i < 3; ++i) {
      // Update time since last heartbeat.
      const now = new Date();
      const time_since =
        container.getElementsByClassName(`p${i}-time-since-heartbeat`)[0];
      time_since.textContent =
        ((now - this.heartbeats[i]) / 1000).toFixed(1) + ' s';

      // Update health.
      const health = container.getElementsByClassName(`p${i}-health`)[0];
      if (this.alive[i]) {
        health.textContent = 'alive';
        health.classList.add('alive');
        time_since.classList.add('alive');
        health.classList.remove('dead');
        time_since.classList.remove('dead');
      } else {
        health.textContent = 'dead';
        health.classList.add('dead');
        time_since.classList.add('dead');
        health.classList.remove('alive');
        time_since.classList.remove('alive');
      }

    }

    // Update processes in barrier.
    const in_barrier = container.getElementsByClassName('in-barrier')[0];
    if (in_barrier) {
      in_barrier.textContent = `In barrier = [${this.in_barrier}]`;
    }
  }
}

// Process

class Process {
  constructor(network, options, i) {
    this.network = network;
    this.options = options;
    this.i = i;
    this.alive = true;
    this.live_devices = null;
    this.heartbeat_cancel =
      setIntervalWithJitter(() => this.send_heartbeat(), 3000, 4000);
  }

  receive(msg) {
    const {src, dst, type, payload} = msg;
    switch (type) {
      case 'live_devices':
        if (this.alive) {
          this.live_devices = payload;
        }
        break;
      case 'fail':
        this.fail();
        break;
      default:
        console.log(`Unknown message type ${type}`)
    }
  }

  send_heartbeat() {
    this.network.push({
      src: this.i,
      dst: 'server',
      type: 'heartbeat',
      payload: '❤️',
    })
  }

  send_live_devices() {
    this.network.push({
      src: this.i,
      dst: 'server',
      type: 'live_devices',
      payload: '⚫',
    })
  }

  fail() {
    this.alive = false;
    this.live_devices = null;
    this.heartbeat_cancel();
  }

  update_html(container) {
    const live_devices =
      container.getElementsByClassName(`p${this.i}-live-devices`)[0];
    if (this.options.live_devices) {
      if (this.live_devices == null) {
        live_devices.textContent = 'live processes = 0,1,2';
      } else {
        live_devices.textContent = `live processes = ${this.live_devices}`;
      }
    }

    if (!this.alive) {
      const node = container.getElementsByClassName(`p${this.i}`)[0];
      node.classList.add('failed');

      const ld_button =
        container.getElementsByClassName(`p${this.i}-ld-button`)[0];
      if (ld_button) {
        ld_button.disabled = true;
      }

      const fail_button =
        container.getElementsByClassName(`p${this.i}-fail-button`)[0];
      if (fail_button) {
        fail_button.disabled = true;
      }
    }
  }
}


// Network communication.

function send(container, tall, text, src, dst, after) {
  const msg = document.createElement('div');
  msg.textContent = text;
  msg.classList.add('msg');
  if (tall) {
    msg.classList.add(`${src}_to_${dst}_tall`);
  } else {
    msg.classList.add(`${src}_to_${dst}`);
  }
  msg.addEventListener('animationend', (_) => {
    msg.remove();
    after();
  });
  container.appendChild(msg);
}

// {
//     share_fate: false,
//     live_devices: false,
//     barrier: false,
// }
function init_cluster(id, options) {
  const container = document.getElementById(id);
  container.innerHTML = `
    <div class="server-box">
      <div class="server">Coordination Service</div>
      <div class="server-data">
        <ul>
          <li>Process 0: <span class="p0-time-since-heartbeat alive">0s</span> (<span class="p0-health alive">alive</span>)</li>
          <li>Process 1: <span class="p1-time-since-heartbeat alive">0s</span> (<span class="p1-health alive">alive</span>)</li>
          <li>Process 2: <span class="p2-time-since-heartbeat alive">0s</span> (<span class="p2-health alive">alive</span>)</li>
          <li class="in-barrier">In barrier: []</li>
        </ul>
      </div>
    </div>

    <div class="proc-box p0-box">
      <div class="proc p0">0</div>
      <div class='p0-live-devices'>live processes = 0,1,2</div>
      <button class="p0-ld-button">Call live_processes</button>
      <button class="p0-fail-button" style="visibility: hidden">Fail</button>
    </div>

    <div class="proc-box p1-box">
      <div class="proc p1">1</div>
      <div class='p1-live-devices'>live processes = 0,1,2</div>
      <button class="p1-ld-button">Call live_processes</button>
      <button class="p1-fail-button" style="visibility: hidden">Fail</button>
    </div>

    <div class="proc-box p2-box">
      <div class="proc p2">2</div>
      <div class='p2-live-devices'>live processes = 0,1,2</div>
      <button class="p2-ld-button">Call live_processes</button>
      <button class="p2-fail-button">Fail</button>
    </div>
  `;

  // Create the cluster.
  let network = [];
  let server = new CoordinationService(network, options);
  const processes = [
    new Process(network, options, 0), new Process(network, options, 1),
    new Process(network, options, 2)
  ];

  // Set up the live_devices button.
  for (let i = 0; i < 3; ++i) {
    const button = container.getElementsByClassName(`p${i}-ld-button`)[0];
    if (options.live_devices) {
      button.addEventListener('click', () => processes[i].send_live_devices());
    } else {
      button.remove();
    }
  }

  // Set up the fail button.
  const button = container.querySelectorAll('.p2-fail-button')[0];
  button.addEventListener('click', () => processes[2].fail());

  // Remove live_devices display if needed.
  if (!options.live_devices) {
    for (let i = 0; i < 3; ++i) {
      container.getElementsByClassName(`p${i}-live-devices`)[0].remove();
    }
  }
  if (!options.barrier) {
    container.getElementsByClassName('in-barrier')[0].remove();
  }

  // Periodically process network messages.
  setInterval(() => {
    while (network.length > 0) {
      const msg = network.shift();
      const tall = options.live_devices;
      send(container, tall, msg.payload, `p${msg.src}`, `p${msg.dst}`, () => {
        if (msg.dst == 'server') {
          server.receive(msg);
        } else {
          processes[msg.dst].receive(msg);
        }
      });
    }
  }, 10)

  // Periodically update HTML.
  setInterval(() => {
    server.update_html(container);
    for (let proc of processes) {
      proc.update_html(container);
    }
  }, 50);
}
