(define (problem prob)
  (:domain pr2-blocksworld)
  (:objects
    red green blue
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on green blue)
    (on red green)
    (clear red)
  )
  (:goal
    (and
    (on-table red)
    (on green red)
    (on blue green)
     )))
