class Logging:
    def __init__(self, log_location, col_titles=None):
        self.log_location = log_location
        log_file = open(log_location, 'wt')
        if col_titles:
            log_file.write(col_titles)
        log_file.close()

    def write(self, log_text):
        log_file = open(self.log_location, 'at')
        log_file.write(log_text + '\n')
        log_file.close()
