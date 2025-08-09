import io
import time
import base64
import numpy as np
from typing import Optional, Dict, Any, Tuple

import gymnasium as gym
from gymnasium import spaces
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


FRAME_SIZE = 84


class BekhaziGameEnv(gym.Env):
    """
    Selenium driven env for bekhazi.ca.
    Observation. 84 by 84 gray. Channel first.
    Actions.
      0. no op
      1. jump
      2. crouch
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        headless: bool = True,
        url: str = "https://bekhazi.ca",
        start_id: str = "startButton",
        again_id: str = "playAgain",
        panel_id: str = "highscores",
        canvas_css: str = "#game"
    ):
        super().__init__()
        self.url = url
        self.start_id = start_id
        self.again_id = again_id
        self.panel_id = panel_id
        self.canvas_css = canvas_css

        opts = webdriver.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1200,800")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(options=opts)
        self.wait = WebDriverWait(self.driver, 20)

        self.driver.get(self.url)
        self._ensure_started()

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(3)

        self.last_score = 0
        self.steps = 0
        self.max_steps = 6000

        # Action smoothing and locks
        self.min_action_repeat = 3     # repeat each chosen action for a few env steps
        self._repeat_left = 0
        self._last_action = 0
        self._jump_lock = 0            # while positive, ignore crouch
        self._crouch_lock = 0          # while positive, ignore jump

    # helpers

    def _focus_canvas(self):
        try:
            canvas = self.driver.find_element(By.CSS_SELECTOR, self.canvas_css)
            ActionChains(self.driver).move_to_element(canvas).click().perform()
        except Exception:
            body = self.driver.find_element(By.TAG_NAME, "body")
            ActionChains(self.driver).move_to_element(body).click().perform()

    def _click_if_present(self, by, selector, timeout=6.0) -> bool:
        try:
            btn = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            ActionChains(self.driver).move_to_element(btn).click().perform()
            time.sleep(0.2)
            return True
        except Exception:
            return False

    def _panel_visible(self) -> bool:
        try:
            panel = self.driver.find_element(By.ID, self.panel_id)
            return panel.is_displayed()
        except Exception:
            return False

    def _ensure_started(self):
        self._click_if_present(By.ID, self.start_id, timeout=10.0)
        self._focus_canvas()
        time.sleep(0.3)

    def _restart_if_needed(self):
        if self._panel_visible():
            if self._click_if_present(By.ID, self.again_id, timeout=6.0):
                self._focus_canvas()
                time.sleep(0.3)

    def _read_score(self) -> int:
        try:
            value = self.driver.execute_script("return window.bekhaziScore ?? 0;")
            if isinstance(value, (int, float)):
                return int(value)
        except Exception:
            pass
        try:
            el = self.driver.find_element(By.ID, "scoreDisplay")
            txt = "".join(ch for ch in el.text if ch.isdigit())
            return int(txt) if txt else 0
        except Exception:
            return 0

    def _grab_frame(self) -> np.ndarray:
        """
        Read canvas pixels via toDataURL. Stable size. No CSS jitter.
        """
        png_bytes = None
        try:
            b64 = self.driver.execute_script(
                """
                const sel = arguments[0];
                const c = document.querySelector(sel);
                if (!c) return null;
                const d = c.toDataURL('image/png');
                if (!d) return null;
                return d.split(',')[1];
                """,
                self.canvas_css,
            )
            if b64:
                png_bytes = base64.b64decode(b64)
        except Exception:
            png_bytes = None

        if png_bytes is None:
            try:
                png_bytes = self.driver.get_screenshot_as_png()
            except Exception:
                return np.zeros((1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

        img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((FRAME_SIZE, FRAME_SIZE))
        arr = np.array(img, dtype=np.uint8)[np.newaxis, :, :]
        return arr

    def _game_over(self) -> bool:
        return self._panel_visible()

    # gym api

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._restart_if_needed()
        self.last_score = self._read_score()
        self.steps = 0
        self._repeat_left = 0
        self._last_action = 0
        self._jump_lock = 0
        self._crouch_lock = 0
        obs = self._grab_frame()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # keep focus
        if self.steps % 50 == 0:
            self._focus_canvas()

        # decrement locks
        if self._jump_lock > 0:
            self._jump_lock -= 1
        if self._crouch_lock > 0:
            self._crouch_lock -= 1

        # action repeat and mutual exclusion
        if self._repeat_left > 0:
            a = self._last_action
            self._repeat_left -= 1
        else:
            a = int(action)
            if a == 2 and self._jump_lock > 0:
                a = 0
            if a == 1 and self._crouch_lock > 0:
                a = 0
            self._last_action = a
            self._repeat_left = self.min_action_repeat - 1

        # send key input
        if a == 1:
            ActionChains(self.driver).key_down(Keys.SPACE).pause(0.02).key_up(Keys.SPACE).perform()
            self._jump_lock = 12   # about half a second at our pace
        elif a == 2:
            ActionChains(self.driver).key_down(Keys.ARROW_DOWN).pause(0.12).key_up(Keys.ARROW_DOWN).perform()
            self._crouch_lock = 6  # brief lock so it does not immediately jump

        time.sleep(0.06)

        obs = self._grab_frame()
        score = self._read_score()

        reward = 0.01
        if score > self.last_score:
            reward += 0.1 * (score - self.last_score)
        self.last_score = score

        terminated = self._game_over()
        self.steps += 1
        truncated = self.steps >= self.max_steps

        if terminated:
            reward -= 1.0
            self._restart_if_needed()

        info = {"score": score, "effective_action": a}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass
