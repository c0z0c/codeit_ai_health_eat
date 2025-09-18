import inspect
import os
import datetime
from enum import Enum

class LV(Enum):
    TRACE = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

class LOG:
    __log_file_enable__ = False  # 클래스(static) 변수로 선언
    __log_file_path__ = ""
    __log_file_name__ = ""
    __log_file__ = None

    def __init__(self, level=LV.TRACE, head=""):
        self.__log_level__ = level
        self.__head__ = head
        self.__log_dir__ = ""
        self.set_log_dir(os.getcwd() + "/logs")
        
    def __del__(self):
        try:
            if hasattr(LOG, "__log_file_close__"):
                self.__log_file_close__()
        except Exception:
            pass        

    def __log_file_close__(self):
        if None != LOG.__log_file__:
            try:
                LOG.__log_file__.close()
            except Exception as e:
                print(f"Error closing log file: {e}")
            finally:
                LOG.__log_file__ = None
    
    def __log_file_create__(self):
        if LOG.__log_file_enable__ and None != LOG.__log_file__:
            return
        try:
            LOG.__log_file__ = open(LOG.__log_file_path__ + "/" + LOG.__log_file_name__, "a", encoding="utf-8")
        except Exception as e:
            print(f"Error creating log file: {e}")
            LOG.__log_file__ = None

    def __log_head_datetime__(self):
        now = datetime.datetime.now()
        #return now.strftime("%Y-%m-%d %H:%M:%S")
        return now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def __log_file_line__(self, deep=1):
        frame = inspect.currentframe()
        for _ in range(deep):
            if frame.f_back is not None:
                frame = frame.f_back
            else:
                return "", 0
        filename = frame.f_code.co_filename
        short_filename = filename.split('\\')[-1].split('/')[-1]
        lineno = frame.f_lineno
        funcname = frame.f_code.co_name
        return short_filename, lineno, funcname
    
    def _log_head__(self):
        if self.__head__:
            return f"[{self.__log_head_datetime__()}] {self.__head__}"
        return f"[{self.__log_head_datetime__()}]"

    def __log_file_write__(self, mesgage):
        if self.__log_file__:
            # print("***", "__log_file_write__")
            try:
                self.__log_file__.write(mesgage)
                self.__log_file__.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}")
                
    def set_head(self, head):
        self.__head__ = head
        return self
            
    def set_log_dir(self, path):
        self.__log_dir__ = path
        try:
            if not os.path.exists(self.__log_dir__):
                os.makedirs(self.__log_dir__)
        except Exception as e:
            print(f"Error creating log directory: {e}")
        return self

    def set_file_enable(self, enable=True):
        LOG.__log_file_enable__ = enable
        if LOG.__log_file_enable__:
            now = datetime.datetime.now()
            log_file_path = self.__log_dir__ + "/" + now.strftime("%y/%m/%d")
            if log_file_path != LOG.__log_file_path__:
                self.__log_file_close__()
            LOG.__log_file_path__ = log_file_path
            LOG.__log_file_name__ = now.strftime("%y%m%d.log")
            try:
                if not os.path.exists(LOG.__log_file_path__):
                    os.makedirs(LOG.__log_file_path__)
            except Exception as e:
                print(f"Error creating log file path: {e}")
            self.__log_file_create__()
        else:
            self.__log_file_close__()
        return self

    def log(self, level, *args, **kwargs):
        head = self._log_head__()
        filename, lineno, funcname = self.__log_file_line__(2)
        if not isinstance(level, LV):
            print(f"{head} {filename}:{lineno} level은 LV Enum 타입이어야 합니다.")
            return
        if level.value < self.__log_level__.value:
            return
        message = ' '.join(str(arg) for arg in args)
        log_message = f"{head} {filename}:{lineno} {funcname} {message}"
        print(log_message + "\n", end="", **kwargs)
        # print("***", LOG.__log_file_enable__, LOG.__log_file_path__, LOG.__log_file_name__, LOG.__log_file__)
        if LOG.__log_file_enable__ and self.__log_file__:
            self.__log_file_write__(log_message + "\n")

DLOG = LOG(LV.TRACE, "LOG")
DLOG.set_file_enable(False)

# Example usage
if __name__ == "__main__":
    DLOG.log("테스트 로그입니다.")
    DLOG.log(LV.TRACE, "테스트 로그입니다.")
    DLOG.log(LV.TRACE, f"end of {__file__}")
